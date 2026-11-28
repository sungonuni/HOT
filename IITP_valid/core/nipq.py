"""NIPQ 의사양자화 (Shin et al., "NIPQ: Noise proxy-based Integrated Pseudo-Quantization", CVPR'23) 의 학습용 구현.
qnet.LSQ 가 학습 중에 쓴다. 평가(eval) 때는 qnet.LSQ 가 실제 반올림을 한다.

NIPQ 학습 (LSQ.noise=True)
  y = clamp(x, lo, hi) + 1[lo < x < hi] · (u − ½) · s,   lo = −qn·s, hi = qp·s,  u ~ U(0, 1)
  dy/dx = 1[lo ≤ x ≤ hi]
  dy/ds = −qn (x < lo),  qp (x > hi),  u − ½ (lo < x < hi),  0 (x = lo 또는 hi)
  (경계값 x = lo, hi 에서의 기울기는 PyTorch torch.clamp 와 같게 정했다)
STE 미세조정 (LSQ.noise=False, 학습 마지막 몇 epoch)
  y = round(clamp(x/s, −qn, qp)) · s,  dy/dx = 1[−qn ≤ x/s ≤ qp],  step size s 는 고정

메모리를 줄인 구현 (EFFICIENT=True, 기본)
  backward 용으로 저장하는 것: 원소당 2bit (마스크 2개: lo ≤ x ≤ hi, x ≥ hi 또는 x = lo) + 난수 시드(정수 1개).
  노이즈 u 는 저장하지 않고 backward 에서 같은 시드로 다시 만든다 (노이즈만 다시 계산하는 activation checkpointing).
PyTorch autograd 그대로 (EFFICIENT=False, 메모리 비교·검증용)
  같은 식·같은 노이즈. autograd 가 x, u, 마스크 등을 FP32/bool 텐서로 저장한다.
"""
import torch
from torch.autograd import Function

from . import hot

EFFICIENT = True
_GEN = {}


def new_seed():
    """전역 난수(torch.manual_seed 로 고정됨)에서 노이즈 시드를 뽑음."""
    return int(torch.randint(0, 2 ** 62, ()).item())


def noise(shape, seed, device, dtype):
    """시드로 정해지는 U(0,1) 노이즈. 같은 시드면 forward·backward 에서 같은 값."""
    key = str(device)
    if key not in _GEN:
        _GEN[key] = torch.Generator(device=device)
    return torch.rand(shape, generator=_GEN[key].manual_seed(seed), device=device, dtype=dtype)


def _reduce(t, shape):
    """broadcast 된 기울기를 shape (s 의 모양) 으로 합침."""
    dims = [d for d in range(t.dim()) if shape[d] == 1 and t.shape[d] != 1]
    return t.sum(dims, keepdim=True) if dims else t


class NoiseFn(Function):
    @staticmethod
    def forward(ctx, x, s, qn, qp, seed):
        lo, hi = s * -qn, s * qp
        closed = (x >= lo) & (x <= hi)
        edge = (x >= hi) | (x == lo)  # closed 와 같이 보면: 안쪽 / 경계 / 위 / 아래 4가지
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
            c = noise(shape, seed, gy.device, gy.dtype).sub_(0.5)  # 안쪽: u − ½
            c.masked_fill_(edge, float(qp)).masked_fill_(closed & edge, 0.0)  # 위: qp, 경계: 0
            c.masked_fill_(~(closed | edge), float(-qn))  # 아래: −qn
            gs = _reduce(c.mul_(gy), s.shape)
        return gx, gs, None, None, None


def noise_naive(x, s, qn, qp, seed):
    """NoiseFn 과 같은 식·같은 노이즈, PyTorch autograd 그대로 (기존 NIPQ 구현 방식)."""
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
