"""NIPQ pseudo-quantization (Shin et al., CVPR'23) with memory-efficient backward.

NIPQ:  y = clamp(x, lo, hi) + 1[lo < x < hi] * (u - 1/2) * s,  lo = -qn*s, hi = qp*s, u ~ U(0, 1)
STE:   y = round(clamp(x/s, -qn, qp)) * s, s frozen
EFFICIENT=True saves only two 1-bit masks and a seed; u is regenerated in backward.
"""
import torch
from torch.autograd import Function

from . import hot

EFFICIENT = True
_GEN = {}


def new_seed():
    return int(torch.randint(0, 2 ** 62, ()).item())


def noise(shape, seed, device, dtype):
    """Seeded U(0, 1) noise, identical in forward and backward."""
    key = str(device)
    if key not in _GEN:
        _GEN[key] = torch.Generator(device=device)
    return torch.rand(shape, generator=_GEN[key].manual_seed(seed), device=device, dtype=dtype)


def _reduce(t, shape):
    """Sum a broadcast gradient back to shape."""
    dims = [d for d in range(t.dim()) if shape[d] == 1 and t.shape[d] != 1]
    return t.sum(dims, keepdim=True) if dims else t


class NoiseFn(Function):
    @staticmethod
    def forward(ctx, x, s, qn, qp, seed):
        lo, hi = s * -qn, s * qp
        closed = (x >= lo) & (x <= hi)
        edge = (x >= hi) | (x == lo)  # with closed: inside / boundary / above / below
        y = torch.clamp(x, lo, hi)
        y.add_(noise(x.shape, seed, x.device, x.dtype).sub_(0.5).mul_(s).mul_(closed & ~edge))
        ctx.save_for_backward(hot.pack_bits(closed), hot.pack_bits(edge), s)
        ctx.conf = (x.shape, qn, qp, seed)
        return y

    @staticmethod
    def backward(ctx, gy):
        p_closed, p_edge, s = ctx.saved_tensors
        shape, qn, qp, seed = ctx.conf
        closed = hot.unpack_bits(p_closed, shape)
        gx = gs = None
        if ctx.needs_input_grad[0]:
            gx = torch.where(closed, gy, 0.0)
        if ctx.needs_input_grad[1]:
            edge = hot.unpack_bits(p_edge, shape)
            c = noise(shape, seed, gy.device, gy.dtype).sub_(0.5)  # inside: u - 1/2
            c.masked_fill_(edge, float(qp)).masked_fill_(closed & edge, 0.0)  # above: qp, boundary: 0
            c.masked_fill_(~(closed | edge), float(-qn))  # below: -qn
            gs = _reduce(c.mul_(gy), s.shape)
        return gx, gs, None, None, None


def noise_naive(x, s, qn, qp, seed):
    """Same as NoiseFn but with plain autograd (memory comparison)."""
    lo, hi = s * -qn, s * qp
    inside = ((x > lo) & (x < hi)).to(x.dtype)
    return torch.clamp(x, lo, hi) + (noise(x.shape, seed, x.device, x.dtype) - 0.5) * s * inside


class SteFn(Function):
    @staticmethod
    def forward(ctx, x, s, qn, qp):
        v = x / s
        ctx.save_for_backward(hot.pack_bits((v >= -qn) & (v <= qp)))
        ctx.shape = x.shape
        return v.clamp_(-qn, qp).round_().mul_(s)

    @staticmethod
    def backward(ctx, gy):
        (p,) = ctx.saved_tensors
        return torch.where(hot.unpack_bits(p, ctx.shape), gy, 0.0), None, None, None
