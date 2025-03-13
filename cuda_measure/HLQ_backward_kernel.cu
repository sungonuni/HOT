#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <cuda.h>
#include <ctime>
#include "cuda_fp16.hpp"
#include "cuda_fp16.h"

#include "cuda_runtime.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/layout/matrix.h"

__device__ int stoch_round(float x){
    // pseudo stochastic rounding
    // IEEE 754's FP32 : 1bit sign, 8bit exponent, 23bit mantissa
    // The first 11-bit of mantissa is probability and last 11-bit of mantissa is pseudo random number.
    int sign = x < 0 ? -1 : 1;
    float x_abs = fabs(x);
    int x_int = floor(x_abs);

    unsigned int* x_ptr = reinterpret_cast<unsigned int*>(&x);
    unsigned int x_bits = *x_ptr;
    unsigned int x_mantissa = x_bits & 0x007FFFFF; // Extract mantissa

    unsigned int prob = x_mantissa >> 11;
    unsigned int pseudo_randn = x_mantissa & 0x7FF; // Extract last 11 bits

    int flip = prob <= pseudo_randn ? 1 : 0;
    return sign * (x_int + flip);
}


template<typename scalar_t>
__global__ void fwht_kernel(scalar_t * MatI, scalar_t * MatO){
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    // Calculate indices with padding
    // const int padding = 0;
    // int paddedRowSize = blockDim.x + padding;
    // int sdataIdx = threadIdx.y * paddedRowSize + threadIdx.x;
    // int matIdx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int sdataIdx = ty * blockDim.x + tx;
    int matIdx = by * gridDim.x + bx * blockDim.x * blockDim.y + ty * blockDim.x + tx;

    // Upload each block of MatI to sdata, considering padding
    if (threadIdx.x < blockDim.x) {
        sdata[sdataIdx] = MatI[matIdx];
    }
    __syncthreads();

    // Conduct FWHT for each sdata row, considering the padding
    int h = 1;
    while (h < blockDim.x) {
        int ty = threadIdx.y;
        int tx = threadIdx.x;
        if ((tx / h) % 2 == 0) {
            if (tx + h < blockDim.x) {
                scalar_t x = sdata[ty * blockDim.x + tx];
                scalar_t y = sdata[ty * blockDim.x + (tx + h)];

                sdata[ty * blockDim.x + tx] = x + y;
                sdata[ty * blockDim.x + (tx + h)] = x - y;
            }
        }
        h *= 2;
        __syncthreads();
    }

    if (threadIdx.x < blockDim.x) {
        MatO[matIdx] = sdata[sdataIdx];
    }
    __syncthreads();
}

template<typename scalar_t>
__global__ void find_scale_kernel(scalar_t * ArrayI, scalar_t * ArrayO, int quantBits){
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    // Calculate indices with padding
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int sdataIdx = ty * blockDim.x + tx;
    int matIdx = by * gridDim.x + bx * blockDim.x * blockDim.y + ty * blockDim.x + tx;

    // Upload each block of MatI to sdata, considering padding
    if (threadIdx.x < blockDim.x) {
        sdata[sdataIdx] = ArrayI[matIdx];
    }
    __syncthreads();

    // Find step_size per token
    if (threadIdx.x < blockDim.x) {
        sdata[sdataIdx] = sdata[sdataIdx] / ((1 << (quantBits-1)) - 1);
    }
    __syncthreads();

    if (threadIdx.x < blockDim.x) {
        ArrayO[matIdx] = sdata[sdataIdx];
    }
    __syncthreads();
}

template<typename scalar_t>
__global__ void quant_int4_kernel(scalar_t * MatI, int8_t * MatO, const float scale){
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    // Calculate indices with padding
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int sdataIdx = ty * blockDim.x + tx;
    int matIdx = by * gridDim.x + bx * blockDim.x * blockDim.y + ty * blockDim.x + tx;

    // Upload each block of MatI to sdata, considering padding
    if (threadIdx.x < blockDim.x) {
        sdata[sdataIdx] = MatI[matIdx];
    }
    __syncthreads();

    // Quantization
    sdata[sdataIdx] = std::clamp((int)(stoch_round((sdata[sdataIdx] / scale))), -7, 7);
    __syncthreads();

    // Download sdata to each block of MatI, considering padding and packing
    if (threadIdx.x < blockDim.x) {
        MatO[matIdx] = ((int)sdata[(sdataIdx<<1)+1] << 4) | ((int)sdata[sdataIdx<<1] & 15);
    }
    __syncthreads();
}

template<typename scalar_t>
__global__ void quant_int4_per_token_kernel(scalar_t * MatI, int8_t * MatO, scalar_t * scale){
    extern __shared__ char smem[];
    extern __shared__ char scale_smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);
    scalar_t* scale_sdata = reinterpret_cast<scalar_t*>(scale_smem);

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * blockDim.x + ty;

    int sdataIdx = ty * blockDim.x + tx;
    int matIdx = by * gridDim.x + bx * blockDim.x * blockDim.y + ty * blockDim.x + tx;
    int scale_sdataIdx = ty;
    int scaleIdx = Row;

    // Upload each block of MatI to sdata, considering padding
    if (threadIdx.x < blockDim.x) {
        sdata[sdataIdx] = MatI[matIdx];
        scale_sdata[scale_sdataIdx] = scale[scaleIdx];
    }
    __syncthreads();

    // Quantization
    sdata[sdataIdx] = std::clamp((int)(stoch_round((sdata[sdataIdx] / scale_sdata[scale_sdataIdx]))), -7, 7);
    __syncthreads();

    // Download sdata to each block of MatI, considering padding and packing
    if (threadIdx.x < blockDim.x) {
        MatO[matIdx] = ((int)sdata[(sdataIdx<<1)+1] << 4) | ((int)sdata[sdataIdx<<1] & 15);
    }
    __syncthreads();
}

//TODO: Define a CUTLASS GEMM template and launch a GEMM kernel. Int4 gemm
cudaError_t CutlassSgemmNN_int4(
  const int M,
  const int N,
  const int K,
  const cutlass::int4b_t *A,
  int lda,
  const cutlass::int4b_t *B,
  int ldb,
  int32_t *C,
  int ldc) {

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = int32_t;                 // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::int4b_t;                       // <- data type of elements in input matrix A
using ElementInputB = cutlass::int4b_t;                       // <- data type of elements in input matrix B
using ElementOutput = int32_t;                      // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 128>;  // <- threadblock tile M = 128, N = 256, K = 64
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 128>;  // <- warp tile M = 64, N = 64, K = 64 
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 64>;  // <- MMA Op tile M = 8, N = 8, K = 16 for INT8

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    4,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 3;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;
  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);
    
  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     {A, lda},  // <- reference to matrix A on device
                                     {B, ldb},  // <- reference to matrix B on device
                                     {C, ldc},  // <- reference to matrix C on device
                                     {C, ldc},  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor
  
    Gemm gemm_op;
    cutlass::Status status = gemm_op(arguments);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

template<typename scalar_t>
__global__ void quant_int8_kernel(scalar_t * MatI, int8_t * MatO, const float scale){
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    // Calculate indices with padding
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;


    int sdataIdx = ty * blockDim.x + tx;
    int matIdx = by * gridDim.x + bx * blockDim.x * blockDim.y + ty * blockDim.x + tx;

    // Upload each block of MatI to sdata, considering padding
    if (threadIdx.x < blockDim.x) {
        sdata[sdataIdx] = MatI[matIdx];
    }
    __syncthreads();

    // Download sdata to each block of MatI, considering padding and packing
    if (threadIdx.x < blockDim.x) {
        MatO[matIdx] = sdata[sdataIdx];
    }
    __syncthreads();
   
}

cudaError_t CutlassSgemmNN_int8(
  int M,
  int N,
  int K,
  const int8_t *A,
  int lda,
  const int8_t *B,
  int ldb,
  int32_t *C,
  int ldc) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  using ElementInputA = int8_t;                       // <- data type of elements in input matrix A
  using ElementInputB = int8_t;                       // <- data type of elements in input matrix B
  using ElementOutput = int32_t;  

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using ElementComputeEpilogue = int32_t;
// Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  using CutlassGemm = cutlass::gemm::device::Gemm<ElementInputA,        // Data-type of A matrix
                                                  LayoutInputA,         // Layout of A matrix
                                                  ElementInputB,        // Data-type of B matrix
                                                  LayoutInputB,         // Layout of B matrix
                                                  ElementOutput,        // Data-type of C matrix
                                                  LayoutOutput>;        // Layout of C matrix

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  CutlassGemm::Arguments args({M, N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  //
  // Launch the CUTLASS GEMM kernel.
  //
  
  cutlass::Status status = gemm_operator(args);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

template<typename scalar_t>
__global__ void dequantize_cuda_kernel(const int32_t * gemm, scalar_t * __restrict__ output, const float scale){
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    // Calculate indices with padding
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int sdataIdx = ty * blockDim.x + tx;
    int matIdx = by * gridDim.x + bx * blockDim.x * blockDim.y + ty * blockDim.x + tx;

    // Upload each block of MatI to sdata, considering padding
    if (threadIdx.x < blockDim.x) {
        sdata[sdataIdx] = gemm[matIdx];
    }
    __syncthreads();
    
    // Dequantization
    sdata[sdataIdx] = scale * sdata[sdataIdx];
    __syncthreads();

    // Download sdata to each block of output
    if (threadIdx.x < blockDim.x) {
        output[matIdx] = sdata[sdataIdx];
    }
    __syncthreads();

}

template<typename scalar_t>
__global__ void dequantize_cuda_per_token_kernel(const int32_t * gemm, scalar_t * __restrict__ output, scalar_t * scale){
    extern __shared__ char smem[];
    extern __shared__ char scale_smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);
    scalar_t* scale_sdata = reinterpret_cast<scalar_t*>(scale_smem);

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * blockDim.y + ty;

    int sdataIdx = ty * blockDim.x + tx;
    int matIdx = by * gridDim.x + bx * blockDim.x * blockDim.y + ty * blockDim.x + tx;
    int scale_sdataIdx = ty;
    int scaleIdx = Row;

    // Upload each block of MatI to sdata, considering padding
    if (threadIdx.x < blockDim.x) {
        sdata[sdataIdx] = gemm[matIdx];
        scale_sdata[scale_sdataIdx] = scale[scaleIdx];
    }
    __syncthreads();
    
    // Dequantization
    sdata[sdataIdx] = scale_sdata[scale_sdataIdx] * sdata[sdataIdx];
    __syncthreads();

    // Download sdata to each block of output
    if (threadIdx.x < blockDim.x) {
        output[matIdx] = sdata[sdataIdx];
    }
    __syncthreads();

}

torch::Tensor padding_col(torch::Tensor x, long long int size, int target_size){
    int padding_size = (target_size - size % target_size);
    return torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, padding_size}));
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<double>> backward_cuda(torch::Tensor Go, torch::Tensor W, torch::Tensor X, const int * reorder_idx_go, const int * reorder_idx_X, int H_size, bool per_token){
    cudaError_t result;
    //Go: (L, O), W: (O, I), X: (L, I)
    long long int L = Go.size(0);
    long long int O = Go.size(1);
    long long int I = W.size(1);
    int block_height = 1024/H_size; // block_size.y
    int block_width = H_size; // block_size.x

    //TODO: ready the tensor after qaunt, gemm, dequant
    auto option_quantize = torch::TensorOptions().dtype(torch::kInt8).device(Go.device());
    auto option_gemm = torch::TensorOptions().dtype(torch::kInt32).device(Go.device());
    auto option_dequantize = torch::TensorOptions().dtype(Go.dtype()).device(Go.device());
    dim3 block_size(block_width);
    dim3 block_size2(block_height, block_width);

    size_t shared_memory_bytes = 96 * 1024;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Go.scalar_type(), "backward_cuda", ([&] {
    cudaFuncSetAttribute(
        fwht_kernel<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes);
    }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Go.scalar_type(), "backward_cuda", ([&] {
    cudaFuncSetAttribute(
        find_scale_kernel<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes);
    }));

    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Go.scalar_type(), "backward_cuda", ([&] {
    cudaFuncSetAttribute(
        quant_int4_kernel<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes);
    }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Go.scalar_type(), "backward_cuda", ([&] {
    cudaFuncSetAttribute(
        quant_int4_per_token_kernel<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes);
    }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Go.scalar_type(), "backward_cuda", ([&] {
    cudaFuncSetAttribute(
        quant_int8_kernel<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes);
    }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Go.scalar_type(), "backward_cuda", ([&] {
    cudaFuncSetAttribute(
        dequantize_cuda_kernel<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes);
    }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Go.scalar_type(), "backward_cuda", ([&] {
    cudaFuncSetAttribute(
        dequantize_cuda_per_token_kernel<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes);
    }));

    // fwht
    torch::Tensor H_Gogi = torch::empty({L, O}, option_dequantize);
    torch::Tensor H_W = torch::empty({O, I}, option_dequantize);
    torch::Tensor GoT = Go.transpose(0,1);
    torch::Tensor H_Gogw = torch::empty({O, L}, option_dequantize);
    torch::Tensor H_X = torch::empty({L, I}, option_dequantize);

    cudaDeviceSynchronize();
    clock_t time_fwht_start = clock();

    dim3 grid_size_Go((O + block_width - 1) / block_width, (L + block_height - 1) / block_height);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Go.scalar_type(), "backward_cuda", ([&] {
    fwht_kernel<scalar_t><<<grid_size_Go, block_size, shared_memory_bytes>>>(
        Go.data_ptr<scalar_t>(), H_Gogi.data_ptr<scalar_t>());
    }));

    dim3 grid_size_W((I + block_width - 1) / block_width, (O + block_height - 1) / block_height);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(W.scalar_type(), "backward_cuda", ([&] {
    fwht_kernel<scalar_t><<<grid_size_W, block_size, shared_memory_bytes>>>(
        W.data_ptr<scalar_t>(), H_W.data_ptr<scalar_t>());
    }));

    dim3 grid_size_GoT((L + block_width - 1) / block_width, (O + block_height - 1) / block_height);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Go.scalar_type(), "backward_cuda", ([&] {
    fwht_kernel<scalar_t><<<grid_size_GoT, block_size, shared_memory_bytes>>>(
        GoT.data_ptr<scalar_t>(), H_Gogw.data_ptr<scalar_t>());
    }));

    dim3 grid_size_X((I + block_width - 1) / block_width, (L + block_height - 1) / block_height);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "backward_cuda", ([&] {
    fwht_kernel<scalar_t><<<grid_size_X, block_size, shared_memory_bytes>>>(
        X.data_ptr<scalar_t>(), H_X.data_ptr<scalar_t>());
    }));

    clock_t time_fwht_end = clock();
    cudaDeviceSynchronize();


    // quant
    long long int pack_O = O>>1;
    torch::Tensor pack_Gogi = torch::empty({L, pack_O}, option_quantize);
    torch::Tensor pack_Wt = torch::empty({I, pack_O}, option_quantize);
    torch::Tensor Gogw_hq = torch::empty({L, O}, option_quantize);
    torch::Tensor X_hq = torch::empty({L, I}, option_quantize);
    torch::Tensor Wt = H_W.transpose(0,1); // (O, I) -> (I, O)

    torch::Tensor step_size_vector = torch::empty({L, 1}, option_dequantize);
    torch::Tensor max_per_token = std::get<0>(H_Gogi.abs().max(1));

    clock_t time_quant_start = clock();  
    
    dim3 grid_size_mpt(1, (L + block_height - 1) / block_height);
    if (per_token){
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(Go.scalar_type(), "backward_cuda", ([&] {
        find_scale_kernel<scalar_t><<<grid_size_mpt, block_size, shared_memory_bytes>>>(
            max_per_token.data_ptr<scalar_t>(), step_size_vector.data_ptr<scalar_t>(), 8);
        }));

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(Go.scalar_type(), "backward_cuda", ([&] {
        quant_int4_per_token_kernel<scalar_t><<<grid_size_Go, block_size2, shared_memory_bytes>>>(
            H_Gogi.data_ptr<scalar_t>(), pack_Gogi.data_ptr<int8_t>(), step_size_vector.data_ptr<scalar_t>());
        }));
    }
    else{
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(Go.scalar_type(), "backward_cuda", ([&] {
        quant_int4_kernel<scalar_t><<<grid_size_Go, block_size, shared_memory_bytes>>>(
            H_Gogi.data_ptr<scalar_t>(), pack_Gogi.data_ptr<int8_t>(), 1);
        }));
    }
    
    dim3 grid_size_Wt((O + block_width - 1) / block_width, (I + block_height - 1) / block_height);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(W.scalar_type(), "backward_cuda", ([&] {
    quant_int4_kernel<scalar_t><<<grid_size_Wt, block_size, shared_memory_bytes>>>(
        Wt.data_ptr<scalar_t>(), pack_Wt.data_ptr<int8_t>(), 1);
    }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Go.scalar_type(), "backward_cuda", ([&] {
    quant_int8_kernel<scalar_t><<<grid_size_GoT, block_size, shared_memory_bytes>>>(
        H_Gogw.data_ptr<scalar_t>(), Gogw_hq.data_ptr<int8_t>(), 1);
    }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "backward_cuda", ([&] {
    quant_int8_kernel<scalar_t><<<grid_size_X, block_size, shared_memory_bytes>>>(
        X.data_ptr<scalar_t>(), X_hq.data_ptr<int8_t>(), 1);
    }));

    clock_t time_quant_end = clock();
    cudaDeviceSynchronize();


    // Low-rank
    clock_t time_low_rank_start = clock();

    torch::Tensor Gogw_hq_lr = Gogw_hq.index({torch::indexing::Slice(0, int(L/2), torch::indexing::None)}).transpose(0,1); // (O, L/2)
    torch::Tensor Xt_hq_lr = X_hq.index({torch::indexing::Slice(0, int(L/2), torch::indexing::None)}).transpose(0,1); // (I, L/2), test needed

    clock_t time_low_rank_end = clock();
    cudaDeviceSynchronize();

    printf("===== HLA check =====\n");
    std::cout << "Gogw_hq: " << Gogw_hq.size(0) << " " << Gogw_hq.size(1) << std::endl;
    std::cout << "Gogw_hq_lr: " << Gogw_hq_lr.size(0) << " " << Gogw_hq_lr.size(1) << std::endl;
    std::cout << "X_hq: " << X_hq.size(0) << " " << X_hq.size(1) << std::endl;
    std::cout << "Xt_hq_lr: " << Xt_hq_lr.size(0) << " " << Xt_hq_lr.size(1) << std::endl;
    printf("\n");

    printf("===== GEMM tensor shape check =====\n");
    std::cout << "pack_Gogi: " << pack_Gogi.size(0) << " " << pack_Gogi.size(1) << std::endl;
    std::cout << "pack_Wt: " << pack_Wt.size(0) << " " << pack_Wt.size(1) << std::endl;
    std::cout << "Gogw_hq_lr: " << Gogw_hq_lr.size(0) << " " << Gogw_hq_lr.size(1) << std::endl;
    std::cout << "Xt_hq_lr: " << Xt_hq_lr.size(0) << " " << Xt_hq_lr.size(1) << std::endl;
    printf("\n");

    torch::Tensor pack_Gogi_dummy = torch::zeros({L, pack_O}, option_quantize);
    torch::Tensor pack_Wt_dummy = torch::zeros({I, pack_O}, option_quantize);
    torch::Tensor Gogw_hq_dummy = torch::zeros({L, O}, option_quantize);
    torch::Tensor X_hq_dummy = torch::zeros({L, I}, option_quantize);

    torch::Tensor Gogw_hq_lr_dummy = Gogw_hq_dummy.index({torch::indexing::Slice(0, int(L/2), torch::indexing::None)}).transpose(0,1); // (O, L/2)
    torch::Tensor Xt_hq_lr_dummy = X_hq_dummy.index({torch::indexing::Slice(0, int(L/2), torch::indexing::None)}).transpose(0,1); // (I, L/2), test needed
    long long int low_rank = Gogw_hq_lr_dummy.size(1);

    if (pack_O % 64 != 0) {
        pack_Gogi_dummy = padding_col(pack_Gogi_dummy, pack_O, 64);
        pack_Wt_dummy = padding_col(pack_Wt_dummy, pack_O, 64);
        pack_O = pack_Gogi_dummy.size(1);
    }
    
    if (low_rank % 8 != 0) {
        Gogw_hq_lr_dummy = padding_col(Gogw_hq_lr_dummy, low_rank, 8);
        Xt_hq_lr_dummy = padding_col(Xt_hq_lr_dummy, low_rank, 8);
        low_rank = Gogw_hq_lr_dummy.size(1);
    }

    printf("===== Padded GEMM tensor shape check =====\n");
    std::cout << "pack_Gogi_dummy: " << pack_Gogi_dummy.size(0) << " " << pack_Gogi_dummy.size(1) << std::endl;
    std::cout << "pack_Wt_dummy: " << pack_Wt_dummy.size(0) << " " << pack_Wt_dummy.size(1) << std::endl;
    std::cout << "Gogw_hq_lr_dummy: " << Gogw_hq_lr_dummy.size(0) << " " << Gogw_hq_lr_dummy.size(1) << std::endl;
    std::cout << "Xt_hq_lr_dummy: " << Xt_hq_lr_dummy.size(0) << " " << Xt_hq_lr_dummy.size(1) << std::endl;
    printf("\n");

    // gemm for int4 and int8
    clock_t time_gemm_start = clock();

    torch::Tensor gx_gemm = torch::empty({L, I}, option_gemm);
    torch::Tensor gw_gemm = torch::empty({O, I}, option_gemm);
    result = CutlassSgemmNN_int4(L, I, pack_O, reinterpret_cast<cutlass::int4b_t *>(pack_Gogi_dummy.data_ptr<int8_t>()), pack_O, 
            reinterpret_cast<cutlass::int4b_t *>(pack_Wt_dummy.data_ptr<int8_t>()), pack_O, gx_gemm.data_ptr<int32_t>(), I);
    // result = CutlassSgemmNN_int8(L, I, pack_O, pack_Gogi.data_ptr<int8_t>(), pack_O, 
    //         pack_Wt.data_ptr<int8_t>(), pack_O, gx_gemm.data_ptr<int32_t>(), I);
    result = CutlassSgemmNN_int8(O, I, low_rank, Gogw_hq_lr_dummy.data_ptr<int8_t>(), low_rank, 
            Xt_hq_lr_dummy.data_ptr<int8_t>(), low_rank, gw_gemm.data_ptr<int32_t>(), I);
    // result = CutlassSgemmNN_int8(O, I, low_rank, Gogw_hq_lr.data_ptr<int8_t>(), low_rank, 
    //      Xt_hq_lr.data_ptr<int8_t>(), low_rank, gw_gemm.data_ptr<int32_t>(), I);

    clock_t time_gemm_end = clock();
    cudaDeviceSynchronize();


    // Final dequantize
    torch::Tensor gx = torch::empty({L, I}, option_dequantize);
    torch::Tensor gw = torch::empty({O, I}, option_dequantize);
    
    printf("===== Result tensor shape check ===== \n");
    std::cout << "gx_gemm: " << gx_gemm.size(0) << " " << gx_gemm.size(1) << std::endl;
    std::cout << "gx: " << gx.size(0) << " " << gx.size(1) << std::endl;
    std::cout << "gw_gemm: " << gw_gemm.size(0) << " " << gw_gemm.size(1) << std::endl;
    std::cout << "gw: " << gw.size(0) << " " << gw.size(1) << std::endl;
    printf("\n");

    cudaDeviceSynchronize();
    clock_t time_dequantize_start = clock();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(gx.scalar_type(), "backward_cuda", ([&] {
    dequantize_cuda_kernel<scalar_t><<<grid_size_X, block_size, shared_memory_bytes>>>(
        gx_gemm.data_ptr<int32_t>(), 
        gx.data_ptr<scalar_t>(),
        1);
    }));

    if (per_token){
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(gx.scalar_type(), "backward_cuda", ([&] {
        dequantize_cuda_per_token_kernel<scalar_t><<<grid_size_W, block_size2, shared_memory_bytes>>>(
            gw_gemm.data_ptr<int32_t>(), 
            gw.data_ptr<scalar_t>(),
            step_size_vector.data_ptr<scalar_t>());
        }));
    }
    else{
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(gx.scalar_type(), "backward_cuda", ([&] {
        dequantize_cuda_kernel<scalar_t><<<grid_size_W, block_size, shared_memory_bytes>>>(
            gw_gemm.data_ptr<int32_t>(), 
            gw.data_ptr<scalar_t>(),
            1);
        }));
    }

    clock_t time_dequantize_end = clock();
    cudaDeviceSynchronize();

    double fwht_time = (double)(time_fwht_end - time_fwht_start) / CLOCKS_PER_SEC;
    double quant_time = (double)(time_quant_end - time_quant_start) / CLOCKS_PER_SEC;
    double low_rank_time = (double)(time_low_rank_end - time_low_rank_start) / CLOCKS_PER_SEC;
    double gemm_time = (double)(time_gemm_end - time_gemm_start) / CLOCKS_PER_SEC;
    double dequantize_time = (double)(time_dequantize_end - time_dequantize_start) / CLOCKS_PER_SEC;

    std::cout << "fwht_time: " << fwht_time << std::endl;
    std::cout << "quant_time: " << quant_time << std::endl;
    std::cout << "low_rank_time: " << low_rank_time << std::endl;
    std::cout << "gemm_time: " << gemm_time << std::endl;
    std::cout << "dequantize_time: " << dequantize_time << std::endl;
  
    std::vector<double> time_vector;
    time_vector.push_back(fwht_time);
    time_vector.push_back(quant_time);
    time_vector.push_back(gemm_time);
    time_vector.push_back(dequantize_time);

    // return output;
    return std::make_tuple(gx, gw, time_vector);
}