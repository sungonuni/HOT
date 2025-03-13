#include <torch/extension.h>

// return gi, gw, time vector
std::tuple<torch::Tensor, torch::Tensor, std::vector<double>> backward_cuda(torch::Tensor Go, torch::Tensor W, torch::Tensor X, const int * reorder_idx_go, const int * reorder_idx_X, int H_size, bool per_token);

std::tuple<torch::Tensor, torch::Tensor, std::vector<double>> backward(torch::Tensor Go, torch::Tensor W, torch::Tensor X, const int * reorder_idx_go, const int * reorder_idx_X, int H_size, bool per_token){
    TORCH_CHECK(Go.type().is_cuda(), "Go must be a CUDA tensor!");
    TORCH_CHECK(Go.is_contiguous(), "Go must be contiguous!");
    TORCH_CHECK(Go.dim() == 2, "Go must be 2D!");

    TORCH_CHECK(W.type().is_cuda(), "Wt must be a CUDA tensor!");
    TORCH_CHECK(W.is_contiguous(), "Wt must be contiguous!");
    TORCH_CHECK(W.dim() == 2, "Wt must be 2D!");

    TORCH_CHECK(X.type().is_cuda(), "X must be a CUDA tensor!");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous!");
    TORCH_CHECK(X.dim() == 2, "X must be 2D!");

    return backward_cuda(Go, W, X, reorder_idx_go, reorder_idx_X, H_size, per_token);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("backward", &backward);
}