"""HOT (Kim et al., arXiv:2503.21261) INT8 backward as autograd Functions; forward stays FP32.

Conv/Linear: save x as INT8 after batch-axis Hadamard; gx and gW are INT8 matmuls in the Hadamard domain.
BatchNorm: save normalized input as per-channel INT8. ReLU: save a 1-bit mask.
INT8 is symmetric [-127, 127] with scale = max|x| / 127.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function

EPS = 1e-11
QMAX = 127
QUANTIZE = True  # False: Hadamard only, used by check_model.py


def hadamard(n):
    lg2 = int(math.log(n, 2)) if n >= 1 else 0
    if 2 ** lg2 != n:
        raise ValueError("n must be int and power of 2.")
    H = np.array([[1]], dtype=np.int64)
    for _ in range(lg2):
        H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))
    return H


def block_size(n):
    """Largest power-of-2 divisor of n, capped at 16."""
    b = 1
    while n % (b * 2) == 0 and b * 2 <= 16:
        b *= 2
    return b


_H = {}


def _H_of(b, device):
    k = (b, str(device))
    if k not in _H:
        _H[k] = torch.tensor(hadamard(b), dtype=torch.float32, device=device)
    return _H[k]


def had(x, dim):
    """Block-diagonal Hadamard along dim; returns (result, b). Applying twice scales by b."""
    b = block_size(x.shape[dim])
    if b == 1:
        return x, 1
    v = x.contiguous().view(math.prod(x.shape[:dim]), x.shape[dim] // b, b, math.prod(x.shape[dim + 1:]))
    return torch.matmul(_H_of(b, x.device), v).view(x.shape), b


def _absmax(x, dim):
    if dim is None:
        lo, hi = torch.aminmax(x)
    else:
        red = [d for d in range(x.dim()) if d != dim]
        lo, hi = x.amin(red), x.amax(red)
    return torch.maximum(lo.neg_(), hi).clamp_min_(EPS)


def _bshape(s, x, dim):
    if dim is None:
        return s
    shape = [1] * x.dim()
    shape[dim] = -1
    return s.view(shape)


def quant(x, dim=None, inplace=False):
    """Symmetric INT8, per-tensor (dim=None) or per-index along dim. inplace reuses x as workspace."""
    if not QUANTIZE:
        return x, torch.ones(() if dim is None else x.shape[dim], device=x.device)
    s = _absmax(x, dim) / QMAX
    y = x.div_(_bshape(s, x, dim)) if inplace else x / _bshape(s, x, dim)
    return y.round_().clamp_(-QMAX, QMAX).to(torch.int8), s


def dequant(q, s, dim=None):
    return q.to(torch.float32, copy=True).mul_(_bshape(s, q, dim))


def qdq(x, dim=None, inplace=False):
    """Quantize-dequantize directly in FP32."""
    if not QUANTIZE:
        return x
    s = _absmax(x, dim) / QMAX
    sb = _bshape(s, x, dim)
    y = x.div_(sb) if inplace else x / sb
    return y.round_().clamp_(-QMAX, QMAX).mul_(sb)


_SHIFT = None


def pack_bits(m):
    global _SHIFT
    flat = m.reshape(-1)
    pad = (-flat.numel()) % 8
    if pad:
        flat = F.pad(flat, (0, pad))
    if _SHIFT is None or _SHIFT.device != m.device:
        _SHIFT = torch.arange(8, dtype=torch.uint8, device=m.device)
    return (flat.view(-1, 8).to(torch.uint8) << _SHIFT).sum(1, dtype=torch.uint8)


def unpack_bits(p, shape):
    n = math.prod(shape)
    return ((p.unsqueeze(1) >> _SHIFT) & 1).view(-1)[:n].view(shape).view(torch.bool)


class ConvFn(Function):
    @staticmethod
    def forward(ctx, x, w, bias, stride, padding, dilation):
        y = F.conv2d(x, w, bias, stride, padding, dilation)
        xb, bb = had(x, 0)
        qx, sx = quant(xb, 0, inplace=bb > 1)
        del xb
        ctx.save_for_backward(qx, sx, w)
        ctx.conf = (x.shape, bb, stride, padding, dilation, bias is not None)
        return y

    @staticmethod
    def backward(ctx, gy):
        qx, sx, w = ctx.saved_tensors
        x_shape, bb, stride, padding, dilation, has_bias = ctx.conf
        gx = gw = gb = None
        if ctx.needs_input_grad[0]:
            gyo, bo = had(gy, 1)
            wo, _ = had(w, 0)
            gyo = qdq(gyo, 0, inplace=bo > 1)
            gx = torch.nn.grad.conv2d_input(x_shape, qdq(wo, inplace=bo > 1), gyo, stride, padding, dilation)
            del gyo
            if bo > 1:
                gx.div_(bo)
        if ctx.needs_input_grad[1]:
            gyb, _ = had(gy, 0)
            gyb = qdq(gyb, 0, inplace=bb > 1)
            gw = torch.nn.grad.conv2d_weight(dequant(qx, sx, 0), w.shape, gyb, stride, padding, dilation)
            del gyb
            if bb > 1:
                gw.div_(bb)
        if has_bias and ctx.needs_input_grad[2]:
            gb = gy.sum((0, 2, 3))
        return gx, gw, gb, None, None, None


class LinearFn(Function):
    @staticmethod
    def forward(ctx, x, w, bias):
        y = F.linear(x, w, bias)
        xb, bb = had(x, 0)
        qx, sx = quant(xb, 0, inplace=bb > 1)
        del xb
        ctx.save_for_backward(qx, sx, w)
        ctx.conf = (bb, bias is not None)
        return y

    @staticmethod
    def backward(ctx, gy):
        qx, sx, w = ctx.saved_tensors
        bb, has_bias = ctx.conf
        gx = gw = gb = None
        if ctx.needs_input_grad[0]:
            gyo, bo = had(gy, 1)
            wo, _ = had(w, 0)
            gx = (qdq(gyo, 0, inplace=bo > 1) @ qdq(wo, inplace=bo > 1)).div_(bo)
        if ctx.needs_input_grad[1]:
            gyb, _ = had(gy, 0)
            gw = (qdq(gyb, 0, inplace=bb > 1).t() @ dequant(qx, sx, 0)).div_(bb)
        if has_bias and ctx.needs_input_grad[2]:
            gb = gy.sum(0)
        return gx, gw, gb


class BNFn(Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, running_mean, running_var, momentum, eps):
        dims = (0, 2, 3)
        n = x.numel() // x.shape[1]
        v = lambda t: t.view(1, -1, 1, 1)  # noqa: E731
        var, mean = torch.var_mean(x, dims, correction=0)
        invstd = torch.rsqrt(var + eps)
        xh = (x - v(mean)).mul_(v(invstd))
        y = torch.addcmul(v(beta), xh, v(gamma))
        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(mean, alpha=momentum)
            running_var.mul_(1 - momentum).add_(var * (n / (n - 1)), alpha=momentum)
        qh, sh = quant(xh, 1, inplace=True)
        del xh
        ctx.save_for_backward(qh, sh, invstd, gamma)
        return y

    @staticmethod
    def backward(ctx, gy):
        qh, sh, invstd, gamma = ctx.saved_tensors
        dims = (0, 2, 3)
        n = gy.numel() // gy.shape[1]
        v = lambda t: t.view(1, -1, 1, 1)  # noqa: E731
        xh = dequant(qh, sh, 1)
        gbeta = gy.sum(dims)
        ggamma = (gy * xh).sum(dims)
        # gx = gamma/sigma * (gy - mean(gy) - xhat * mean(gy * xhat))
        gx = gy - v(gbeta / n)
        gx.sub_(xh.mul_(v(ggamma / n)))
        del xh
        gx.mul_(v(gamma * invstd))
        return gx, ggamma, gbeta, None, None, None, None


class ReLUFn(Function):
    @staticmethod
    def forward(ctx, x):
        y = x.clamp_min(0)
        ctx.save_for_backward(pack_bits(y > 0))
        ctx.shape = y.shape
        return y

    @staticmethod
    def backward(ctx, gy):
        (p,) = ctx.saved_tensors
        return torch.where(unpack_bits(p, ctx.shape), gy, 0.0)

