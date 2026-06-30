import torch
import torch.nn as nn
import torch.nn.functional as F
import args

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# tensorcast (tcast) provides FP4 (e2m1) / MXFP4 fake quantization.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "tensorcast"))
import tcast

from torch.autograd.function import Function
from utils.hadamard_matmul import *
from utils.utils import batch_z_score_threshold
from torch import Tensor
import torch.cuda.nvtx as nvtx


TD = Transform_Dict()

def norm_quantizer_BWD(x, quantBits):
    mx = torch.max(torch.abs(x), dim=1)[0].unsqueeze(dim=1)
    s = mx / (2**(quantBits-1)-1)
    x = torch.clamp(x, -mx, mx) / (s+1e-11)

    sign = torch.sign(x)
    x_int = torch.floor(x.abs())
    x_decimal = x.abs() - x_int

    flip = torch.bernoulli(x_decimal)

    x_int = sign * (x_int + flip) 
    return x_int * s

def norm_quantizer(x, quantBits, is_batched):
    if is_batched: # if batched
        norm_quantizer_BWD_vmap = torch.vmap(norm_quantizer_BWD, in_dims=(0, None), randomness='different')
        return norm_quantizer_BWD_vmap(x, quantBits), 1.
    return norm_quantizer_BWD(x, quantBits), 1.


def sawb_quantizer_BWD(x, quantBits):
    c1 = 12.1
    c2 = 12.2
    clip = (c1*torch.sqrt(torch.mean(x**2))) - (c2*torch.mean(x.abs()))
    s = clip / (2**(quantBits-1)-1)

    x = torch.clamp(x, -clip, clip) / (s+args.eps)
    x_int = torch.round(x)

    return x_int, s

def sawb_quantizer(x, quantBits, is_batched):
    if is_batched: # if batched
        sawb_quantizer_BWD_vmap = torch.vmap(sawb_quantizer_BWD, in_dims=(0, None), randomness='different')
        return sawb_quantizer_BWD_vmap(x, quantBits)
    return sawb_quantizer_BWD(x, quantBits)


def luq_quantizer_BWD(x, quantBits):

    mx = torch.max(x)
    s = mx / (2**(2**(quantBits-1)-1))
    s_eps = s * torch.rand(x.shape, device=x.device)

    x_abs = x.abs()

    out = torch.where(x_abs < s, s*torch.sign(x), x)
    out = torch.where(x_abs < s_eps, torch.tensor([0], dtype=torch.float32, device=x.device), out)

    out_q = out.clone()

    noise = (2 ** torch.floor(torch.log2((out_q.abs() / s)) )) * out_q.new_zeros(out_q.shape).uniform_(-0.5,0.5)
    out_q = 2 ** torch.floor(torch.log2(((out_q.abs() / s) + noise) * 4/3 ))

    out_q = torch.sign(out) * torch.where(out_q < (2 ** torch.floor(torch.log2(((out.abs() / s))))), (2 ** torch.floor(torch.log2(((out.abs()/s))))), out_q)
    out_q = torch.where(out == 0, torch.tensor([0], dtype=torch.float, device=x.device), out_q)

    return out_q, s

def luq_quantizer(x, quantBits, is_batched):
    if is_batched: # if batched
        luq_qauntizer_BWD_vmap = torch.vmap(luq_quantizer_BWD, in_dims=(0, None), randomness='different')
        return luq_qauntizer_BWD_vmap(x, quantBits)
    return luq_quantizer_BWD(x, quantBits)


def stoch_quantizer_BWD(x, quantBits):
    
    mx = x.abs().max()
    s = mx / (2**(quantBits-1)-1)
    x = torch.clamp(x, -mx, mx) / (s+args.eps)

    sign = torch.sign(x)
    x_int = torch.floor(x.abs())
    x_decimal = x.abs() - x_int

    flip = torch.bernoulli(x_decimal)

    x_int = sign * (x_int + flip)

    return x_int, s

def stoch_quantizer(x, quantBits, is_batched):
    if is_batched: # if batched
        stoch_quantizer_vmap = torch.vmap(stoch_quantizer_BWD, in_dims=(0, None), randomness='different')
        return stoch_quantizer_vmap(x, quantBits)
    return stoch_quantizer_BWD(x, quantBits)

def int_quantizer_BWD(x, quantBits):
    '''
    original
    '''
    mx = x.abs().max()
    s = mx / (2**(quantBits-1)-1)
    x = torch.clamp(x, -mx, mx) / (s+args.eps)
    x_int = torch.round(x)
    return x_int, s

def int_quantizer(x, quantBits, is_batched):
    if is_batched: # if batched
        int_quantizer_vmap = torch.vmap(int_quantizer_BWD, in_dims=(0, None), randomness='different')
        return int_quantizer_vmap(x, quantBits)
    return int_quantizer_BWD(x, quantBits)

def cluster_quant(x, max_per_token, threshold, quantBits):
    '''
    2d tensor, NOT OPTIMIZED
    '''

    # build mask
    x_above_mask = torch.where(max_per_token > threshold, 1, 0).unsqueeze(dim=1)
    x_under_mask = torch.where(max_per_token <= threshold, 1, 0).unsqueeze(dim=1)

    # above quant
    x_above = x * x_above_mask
    x_above_q, s_above_q = stoch_quantizer_BWD(x_above, quantBits)

    # under quant
    x_under = x * x_under_mask
    x_under_q, s_under_q = stoch_quantizer_BWD(x_under, quantBits)

    return (x_above_q * s_above_q) + (x_under_q * s_under_q)


def cluster_quantizer_BWD(x, quantBits):
    '''
    3d tensor
    '''
    mean_per_token = torch.mean(torch.abs(x), dim=2)
    threshold = batch_z_score_threshold(mean_per_token, threshold=1.2)

    cluster_quant_vmap = torch.vmap(cluster_quant, in_dims=(0, 0, 0, None), randomness='different')
    return cluster_quant_vmap(x, mean_per_token, threshold, quantBits)

def cluster_quantizer(x, quantBits, is_batched):
    if is_batched:
        return cluster_quantizer_BWD(x, quantBits), 1.
    else:
        return stoch_quantizer_BWD(x, quantBits)

quant_scheme_dict = {
    "int_quantizer": int_quantizer,
    "stoch_quantizer": stoch_quantizer,
    "sawb_quantizer": sawb_quantizer,
    "luq_quantizer": luq_quantizer,
    "norm_quantizer": norm_quantizer,
}

# ---------------------------------------------------------------------------
# FP4 / MXFP4 fake quantization via tensorcast (tcast)
# ---------------------------------------------------------------------------
# tcast performs *virtual* (fake) quantization: it returns a float tensor that
# has been quantized to the target format and then rescaled back to float.
# Unlike the INT quantizers above there is NO separate integer tensor + scale,
# so the returned scale is always 1.0 (the rescale is baked into the tensor).
# This lets a tensor quantized to mxfp4/fp4 be mixed freely in the same matmul
# with an INT-quantized tensor: ``(a @ b) * scale_a * scale_b`` still holds.
#
#   mxfp4 : OCP MXFP4  -> e2m1 elements sharing one e8m0 power-of-two scale per
#                         32-element tile along the last (contraction) dim.
#   fp4   : per-tensor -> e2m1 elements with a single e8m0 scale for the whole
#                         tensor (no tile-size divisibility constraint).
#
# Stochastic rounding is intentionally NOT applied to FP formats: rounding is
# handled internally by tcast (default round-to-nearest-even).
FP_QUANT_SCHEMES = ("mxfp4", "fp4")
INT8_STORABLE_SCHEMES = ("int", "stoch", "sawb")  # produce int-valued tensors that fit torch.int8

_tcast_dtype_cache = {}

def _get_tcast_dtype(fp_format):
    if fp_format not in _tcast_dtype_cache:
        if fp_format == "mxfp4":
            _tcast_dtype_cache[fp_format] = tcast.mxfp4e2                                  # e2m1fnuz + e8m0_t32
        elif fp_format == "fp4":
            _tcast_dtype_cache[fp_format] = tcast.datatype("e2m1fnuz", "e8m0", "fp4_e8m0")  # per-tensor e8m0
        else:
            raise Exception('FP quant format not implemented: ' + str(fp_format))
    return _tcast_dtype_cache[fp_format]

def fp_quantizer(x, fp_format):
    '''tensorcast fake-quant. Returns (quantized_float_tensor, scale=1.0).'''
    dtype = _get_tcast_dtype(fp_format)
    # MXFP4 tiles the last dim in groups of 32. If it isn't a multiple of 32,
    # zero-pad up to the next multiple, quantize, then slice back. Padding zeros
    # don't change any tile's max-abs (the per-tile e8m0 scale), so the real
    # elements get exactly their proper tile scale.
    if fp_format == "mxfp4":
        tile = 32
        last = x.shape[-1]
        pad = (-last) % tile
        if pad:
            x = F.pad(x, (0, pad))
            xq = tcast.vcast(x, dtype).clone()
            return xq[..., :last], 1.0
    # tcast.cast runs under @torch.inference_mode, so vcast returns an inference
    # tensor that cannot be saved for backward. Clone it to a normal tensor.
    xq = tcast.vcast(x, dtype).clone()
    return xq, 1.0

def quantize_tensor(x, scheme, quantBits, is_batched):
    '''Dispatch quantization by scheme.

    INT-family schemes return (int-valued float tensor, scale) for an integer
    GEMM that is rescaled afterwards.  FP/MXFP schemes use tcast fake-quant and
    return (float tensor, 1.0).'''
    if scheme in FP_QUANT_SCHEMES:
        return fp_quantizer(x, scheme)
    elif scheme == 'int':
        return int_quantizer(x, quantBits, is_batched)
    elif scheme == 'stoch':
        return stoch_quantizer(x, quantBits, is_batched)
    elif scheme == 'sawb':
        return sawb_quantizer(x, quantBits, is_batched)
    elif scheme == 'luq':
        return luq_quantizer(x, quantBits, is_batched)
    elif scheme == 'nq':
        return norm_quantizer(x, quantBits, is_batched)
    elif scheme == 'no' or scheme == False:
        return x, 1.0
    else:
        raise Exception('Quant scheme not implemented: ' + str(scheme))

def _check_gemm_mode():
    '''realGEMM (Triton GEMM) is not implemented yet; only fakeGEMM is supported.'''
    if getattr(args, 'realGEMM', False):
        raise NotImplementedError(
            "realGEMM (Triton GEMM backward) is not implemented yet. "
            "TODO: dispatch to the Triton low-precision GEMM kernels here. "
            "Set --realGEMM false / --fakeGEMM true to use tcast fake quantization.")

# INT Matmul via F.Linear
class FLinearQ(Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x, w, h_out, h_bs, name):
        fin_output = F.linear(x, w)
        return fin_output

    @staticmethod
    def setup_context(ctx, inputs, output):
        x = inputs[0]
        is_batched_x = True if len(x.shape) == 3 else False
        h_bs = inputs[3]
        if args.TransformInput == True:
            if args.transform_scheme == "hadamard":
                x = lowering_matmul_back(h_bs, x)
            elif args.transform_scheme == "low_rank":
                x = lowering_matmul_back(h_bs, x)
            elif args.transform_scheme == "gilr_gwh":
                x = lowering_matmul_back(h_bs.mT, x)
            elif args.transform_scheme == "gih_gwlr":
                x = lowering_matmul_back(h_bs, x)
            else:
                raise Exception('Activation transform scheme not implemented: ' + args.transform_scheme)
        elif args.TransformInput == False:
            None
        
        _check_gemm_mode()
        x, scale_x = quantize_tensor(x, args.quantBWDAct, args.actQuantBit, is_batched_x)
        # INT-quantized activations are stored as int8 to save memory; FP4/MXFP4
        # fake-quant stays float (already rescaled) and feeds a float matmul.
        if args.quantBWDAct in INT8_STORABLE_SCHEMES:
            x = x.to(torch.int8)

        # save_for_backward only accepts tensors; FP4/MXFP4 and 'no' return a
        # scalar scale, so materialize it as a tensor (1.0 is a no-op rescale).
        if not torch.is_tensor(scale_x):
            scale_x = torch.tensor([float(scale_x)], device=x.device)

        ctx.name = inputs[4]
        ctx.save_for_backward(x, inputs[1], inputs[2], inputs[3], scale_x)

    @staticmethod
    def backward(ctx, grad_output):
        _check_gemm_mode()
        layer_name = ctx.name
        x, w, h_out, h_bs, scale_x = ctx.saved_tensors
        if w.dtype == torch.bfloat16:
            h_out = h_out.to(torch.bfloat16).to(w.device)
            h_bs = h_bs.to(torch.bfloat16).to(w.device)
        # import pdb; pdb.set_trace()

        is_batched_x = True if len(x.shape) == 3 else False
        is_batched_w = True if len(w.shape) == 3 else False
        is_batched_grad_output = True if len(grad_output.shape) == 3 else False

        # Hadamard transformation of grad tensor in FP and Quantization to INT
        if args.TransformGogi == True: 
            if args.transform_scheme == "hadamard":
                grad_output_ho = lowering_matmul_front(grad_output, h_out)
            elif args.transform_scheme == "low_rank":
                grad_output_ho = lowering_matmul_back(h_bs, grad_output)
            elif args.transform_scheme == "gilr_gwh":
                grad_output_ho = lowering_matmul_back(h_out, grad_output) # h_out is WHT matrix(quantBatchSize) in gilr_gwh
            elif args.transform_scheme == "gilro_gwh":
                grad_output_ho = lowering_matmul_front(grad_output, h_out.mT)
            elif args.transform_scheme == "gih_gwlr":
                grad_output_ho = lowering_matmul_front(grad_output, h_out)
            else:
                raise Exception('GradHo transform scheme not implemented: ' + args.transform_scheme)
        elif args.TransformGogi == False :
            grad_output_ho = grad_output
        
        if args.quantAuto and len(args.layer_quant_dict) > 0:
            grad_output_ho, scale_gogi= quant_scheme_dict[args.layer_quant_dict[layer_name+'_gx']](grad_output_ho, args.GogiQuantBit, is_batched_grad_output)
        else:
            grad_output_ho, scale_gogi= quantize_tensor(grad_output_ho, args.quantBWDGogi, args.GogiQuantBit, is_batched_grad_output)

        # Hadamard transformation of weight tensor in FP and Quantization to INT
        if args.TransformWgt == True:
            if args.transform_scheme == "hadamard":
                w = lowering_matmul_back(h_out, w)
            elif args.transform_scheme == "low_rank": 
                None
            elif args.transform_scheme == "gilr_gwh":
                None
            elif args.transform_scheme == "gilro_gwh":
                w = lowering_matmul_back(h_out, w)
            elif args.transform_scheme == "gih_gwlr":
                w = lowering_matmul_back(h_out, w)
            else:
                raise Exception('Weight transform scheme not implemented: ' + args.transform_scheme)
        elif args.TransformWgt == False:
            None
        
        w, scale_w = quantize_tensor(w, args.quantBWDWgt, args.weightQuantBit, is_batched_w)

        # Compute the gradient of activation in INT and clamping
        grad_input = (grad_output_ho @ w) * \
            (scale_gogi.unsqueeze(1).unsqueeze(2) if is_batched_grad_output and torch.is_tensor(scale_gogi) else scale_gogi) * \
            (scale_w.unsqueeze(1).unsqueeze(2) if is_batched_w and torch.is_tensor(scale_w) else scale_w)

        # Hadamard transformation of grad tensor in FP and Quantization to INT
        if args.TransformGogw == True:
            if args.transform_scheme == "hadamard":
                grad_output_hb = lowering_matmul_front(grad_output.mT, h_bs)
            elif args.transform_scheme == "low_rank":
                grad_output_hb = lowering_matmul_back(h_bs, grad_output).mT
            elif args.transform_scheme == "gilr_gwh":
                grad_output_hb = lowering_matmul_front(grad_output.mT, h_bs)
            elif args.transform_scheme == "gih_gwlr":
                grad_output_hb = lowering_matmul_back(h_bs, grad_output).mT
            else:
                raise Exception('GradHb transform scheme not implemented: ' + args.transform_scheme)
        elif args.TransformGogw == False:
            grad_output_hb = grad_output.mT
        
        if args.quantAuto and len(args.layer_quant_dict) > 0:
            grad_output_hb, scale_gogw= quant_scheme_dict[args.layer_quant_dict[layer_name+'_gw']](grad_output_hb, args.GogiQuantBit, is_batched_grad_output)
        else:
            grad_output_hb, scale_gogw = quantize_tensor(grad_output_hb, args.quantBWDGogw, args.GogwQuantBit, is_batched_grad_output)

        grad_w = (grad_output_hb @ x.to(grad_output_hb.dtype)) * \
            (scale_gogw.unsqueeze(1).unsqueeze(2) if is_batched_grad_output and torch.is_tensor(scale_gogw) else scale_gogw) * \
            (scale_x.unsqueeze(1).unsqueeze(2) if is_batched_x and torch.is_tensor(scale_x) else scale_x)

        # Rescale due to transform
        TransformGradInput = args.TransformGogi and args.TransformWgt
        TransformGradWeight = args.TransformGogw and args.TransformInput

        if args.transform_scheme == "hadamard":
            grad_input = grad_input / (biggest_power2_factor_max16(h_out.shape[0]) if TransformGradInput else 1)
            grad_w = grad_w / (biggest_power2_factor_max16(h_bs.shape[0]) if TransformGradWeight else 1)
        elif args.transform_scheme == "low_rank":
            grad_input = (lowering_matmul_back(h_bs.mT, grad_input) / biggest_power2_factor(h_bs.shape[1]) if TransformGradInput else grad_input)
            grad_w = grad_w / (biggest_power2_factor(h_bs.shape[1]) if TransformGradWeight else 1)
        elif args.transform_scheme == "gilr_gwh": # h_out is WHT matrix(quantBatchSize) in gilr_gwh
            grad_input = (lowering_matmul_back(h_out.mT, grad_input) / biggest_power2_factor(h_out.shape[1]) if TransformGradInput else grad_input)
            grad_w = grad_w / (biggest_power2_factor_max16(h_bs.shape[0]) if TransformGradWeight else 1)
        elif args.transform_scheme == "gilro_gwh":
            grad_input = grad_input / biggest_power2_factor(h_out.shape[1]) if TransformGradInput else grad_input
            grad_w = grad_w / (biggest_power2_factor_max16(h_bs.shape[0]) if TransformGradWeight else 1)
        elif args.transform_scheme == "gih_gwlr":
            grad_input = grad_input / (biggest_power2_factor_max16(h_out.shape[0]) if TransformGradInput else 1)
            grad_w = grad_w / (biggest_power2_factor(h_bs.shape[1]) if TransformGradWeight else 1)
        else:
            raise Exception('args.transform_scheme not implemented: ' + args.transform_scheme)

        return grad_input, grad_w, None, None, None, None, None, None, None, None
            
class HQ_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super(HQ_Linear, self).__init__(in_features, out_features, bias)
        if args.transform_scheme == "hadamard":
            self.register_buffer('hadamard_out', torch.tensor(TD.get_or_register('hadamard', self.out_features), dtype=self.weight.dtype))
            self.register_buffer('hadamard_bs', torch.tensor(TD.get_or_register('hadamard', args.BATCH_SIZE), dtype=self.weight.dtype))
        elif args.transform_scheme == "low_rank":
            self.register_buffer('hadamard_out', torch.zeros((1, 1), dtype=self.weight.dtype))
            self.register_buffer('hadamard_bs', torch.tensor(TD.get_or_register('low_rank', args.BATCH_SIZE, args.vectorPercentile), dtype=self.weight.dtype))
        elif args.transform_scheme == "gilr_gwh":
            self.register_buffer('hadamard_out', torch.tensor(TD.get_or_register('low_rank', args.BATCH_SIZE, args.vectorPercentile), dtype=self.weight.dtype))
            self.register_buffer('hadamard_bs', torch.tensor(TD.get_or_register('hadamard', args.BATCH_SIZE), dtype=self.weight.dtype))
        elif args.transform_scheme == "gilro_gwh":
            self.register_buffer('hadamard_out', torch.tensor(TD.get_or_register('low_rank', self.out_features, args.vectorPercentile), dtype=self.weight.dtype))
            self.register_buffer('hadamard_bs', torch.tensor(TD.get_or_register('hadamard', args.BATCH_SIZE), dtype=self.weight.dtype))
        elif args.transform_scheme == "gih_gwlr":
            self.register_buffer('hadamard_out', torch.tensor(TD.get_or_register('hadamard', self.out_features), dtype=self.weight.dtype))
            self.register_buffer('hadamard_bs', torch.tensor(TD.get_or_register('low_rank', args.BATCH_SIZE, args.vectorPercentile), dtype=self.weight.dtype))
        else:
            raise Exception('Transformation method not implemented: ' + args.transform_scheme)
        self.layer_name = ' '

    def forward(self, input: Tensor) -> Tensor:
        # Update h_bs to real batch size
        if len(input.shape) == 2 and args.BATCH_SIZE != input.shape[0]:
            if args.transform_scheme == "hadamard":
                self.hadamard_bs = torch.tensor(TD.get_or_register('hadamard', 
                                                                   input.shape[0]), dtype=self.weight.dtype).to(self.weight.device)
            elif args.transform_scheme == "low_rank":
                self.hadamard_bs = torch.tensor(TD.get_or_register('low_rank', 
                                                                   input.shape[0], args.vectorPercentile), dtype=self.weight.dtype).to(self.weight.device)
            elif args.transform_scheme == "gilr_gwh":
                self.hadamard_out = torch.tensor(TD.get_or_register('low_rank', 
                                                                    input.shape[0], args.vectorPercentile), dtype=self.weight.dtype).to(self.weight.device)
                self.hadamard_bs = torch.tensor(TD.get_or_register('hadamard', 
                                                                   input.shape[0]), dtype=self.weight.dtype).to(self.weight.device)
            elif args.transform_scheme == "gilro_gwh":
                self.hadamard_bs = torch.tensor(TD.get_or_register('hadamard', 
                                                                   input.shape[0]), dtype=self.weight.dtype).to(self.weight.device)
            elif args.transform_scheme == "gih_gwlr":
                self.hadamard_bs = torch.tensor(TD.get_or_register('low_rank', 
                                                                   input.shape[0], args.vectorPercentile), dtype=self.weight.dtype).to(self.weight.device)
            else:
                raise Exception('Transformation method not implemented: ' + args.transform_scheme)
        elif len(input.shape) == 3:
            if args.transform_scheme == "hadamard":
                self.hadamard_bs = torch.tensor(TD.get_or_register('hadamard', 
                                                                   input.shape[1]), dtype=self.weight.dtype).to(self.weight.device)
            elif args.transform_scheme == "low_rank":
                self.hadamard_bs = torch.tensor(TD.get_or_register('low_rank', 
                                                                   input.shape[1], args.vectorPercentile), dtype=self.weight.dtype).to(self.weight.device)
            elif args.transform_scheme == "gilr_gwh":
                self.hadamard_out = torch.tensor(TD.get_or_register('low_rank', 
                                                                    input.shape[1], args.vectorPercentile), dtype=self.weight.dtype).to(self.weight.device)
                self.hadamard_bs = torch.tensor(TD.get_or_register('hadamard', 
                                                                   input.shape[1]), dtype=self.weight.dtype).to(self.weight.device)
            elif args.transform_scheme == "gilro_gwh":
                self.hadamard_bs = torch.tensor(TD.get_or_register('hadamard', 
                                                                   input.shape[1]), dtype=self.weight.dtype).to(self.weight.device)
            elif args.transform_scheme == "gih_gwlr":
                self.hadamard_bs = torch.tensor(TD.get_or_register('low_rank', 
                                                                   input.shape[1], args.vectorPercentile), dtype=self.weight.dtype).to(self.weight.device)
            else:
                raise Exception('Transformation method not implemented: ' + args.transform_scheme)
        else:
            None

        # Matmul via Linear
        output = FLinearQ.apply(input, self.weight, self.hadamard_out, self.hadamard_bs, self.layer_name)
        if self.bias is not None:
            output += self.bias.to(output.device)

        return output


# Conv2d via im2col
class HQ_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros') -> None:
        super(HQ_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        if args.transform_scheme == "hadamard":
            self.register_buffer('hadamard_out', torch.tensor(
                TD.get_or_register('hadamard', self.out_channels), dtype=self.weight.dtype))
        elif args.transform_scheme == "low_rank":
            self.register_buffer('hadamard_out', torch.zeros((1, 1), dtype=self.weight.dtype))
        elif args.transform_scheme == "gilr_gwh":
            self.register_buffer('hadamard_out', torch.zeros((1, 1), dtype=self.weight.dtype))
        elif args.transform_scheme == "gilro_gwh":
            self.register_buffer('hadamard_out', torch.tensor(
                TD.get_or_register('low_rank', self.out_channels), dtype=self.weight.dtype))
        elif args.transform_scheme == "gih_gwlr":
            self.register_buffer('hadamard_out', torch.tensor(
                TD.get_or_register('hadamard', self.out_channels), dtype=self.weight.dtype))
        else:
            raise Exception('Transformation method not implemented: ' + args.transform_scheme)
        
        self.register_buffer('hadamard_bs', torch.zeros((1, 1), dtype=self.weight.dtype))
        self.layer_name = ' '

    def forward(self, input):
        original_input_shape = input.shape

        if self.groups > 1: # if depth-wise convolution
            raise NotImplementedError        

        # im2col
        input = torch.nn.functional.unfold(
            input, self.kernel_size, padding=self.padding, stride=self.stride).transpose(1, 2)
        weight = self.weight.view(self.weight.size(0), -1)

        # Update h_bs to real H size (B, H_unfold, W_unfold)
        if args.transform_scheme == "hadamard":
            self.hadamard_bs = torch.tensor(TD.get_or_register('hadamard', 
                                                               input.shape[1]), dtype=self.weight.dtype).to(self.weight.device)
        elif args.transform_scheme == "low_rank":
            self.hadamard_bs = torch.tensor(TD.get_or_register('low_rank', 
                                                               input.shape[1], args.vectorPercentile), dtype=self.weight.dtype).to(self.weight.device)
        elif args.transform_scheme == "gilr_gwh":
            self.hadamard_out = torch.tensor(TD.get_or_register('low_rank', 
                                                                input.shape[1], args.vectorPercentile), dtype=self.weight.dtype).to(self.weight.device)
            self.hadamard_bs = torch.tensor(TD.get_or_register('hadamard', 
                                                               input.shape[1]), dtype=self.weight.dtype).to(self.weight.device)
        elif args.transform_scheme == "gilro_gwh":
            self.hadamard_bs = torch.tensor(TD.get_or_register('hadamard', 
                                                               input.shape[1]), dtype=self.weight.dtype).to(self.weight.device)    
        elif args.transform_scheme == "gih_gwlr":
            self.hadamard_bs = torch.tensor(TD.get_or_register('low_rank', 
                                                               input.shape[1], args.vectorPercentile), dtype=self.weight.dtype).to(self.weight.device)
            
        output = FLinearQ.apply(input, weight, self.hadamard_out, self.hadamard_bs, self.layer_name)
        
        # Reshape the output into BCHW
        output = output.transpose(1, 2)
        output = output.view((original_input_shape[0], self.out_channels, math.ceil(original_input_shape[-2] / self.stride[0]), math.ceil(original_input_shape[-1] / self.stride[1])))

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output