"""HOT (Hadamard-based Optimized Training, Kim et al., arXiv:2503.21261) 의 INT8 backward (autograd Function).
HOT 원본 코드(utils/hadamard_matmul.py, utils/quant_matmul.py)에서 필요한 부분만 옮겨 다시 썼다. 외부 의존성 없음.
qnet.py 의 층(FusedConv, QAdd, QLinear, QReLU)이 학습 중(hot=True)에 이 함수들을 쓴다.

forward 는 입력(NIPQ 의사양자화를 거친 값) 그대로 FP32 로 계산한다. backward 에 필요한 텐서만 정수로 저장하고,
backward 행렬곱을 INT8 로 한다.

Conv2d / Linear  (y = x * W)
  - 저장: 입력 x 를 배치 방향 Hadamard 변환 후 INT8 로 저장 (FP32 x 는 저장하지 않음)
  - 입력 기울기 gx = gy * W   : gy, W 를 출력 채널 방향 Hadamard 변환 → INT8 → 곱한 뒤 1/블록크기
  - 가중치 기울기 gW = x^T gy : gy 를 배치 방향 Hadamard 변환 → INT8, 저장한 INT8 x 와 곱한 뒤 1/블록크기
  Hadamard 블록 H (크기 b ≤ 16, H·H = b·I) 는 합을 내는 축에 걸기 때문에 양자화 전에는 결과가 정확히 같다.
  변환은 값의 이상치를 여러 원소로 퍼뜨려 INT8 양자화 오차를 줄인다.
BatchNorm2d
  - 저장: 정규화된 입력 x̂ 를 채널별 INT8 로 저장 (+ 채널별 1/σ), backward 는 표준 BN 식
ReLU
  - 저장: 출력 > 0 마스크를 1bit 로 묶어 저장 (8개 → 1바이트)

양자화: 대칭 INT8 [-127, 127], scale = max|x| / 127, 반올림 (HOT int_quantizer 와 같음).
scale 단위: x·gy 는 배치 행(Hadamard 로 섞인 표본)마다, W 는 텐서 전체, x̂ 는 채널마다.
정수 행렬곱은 정수 값을 FP32 로 복원해 PyTorch conv/matmul 로 계산한다 (값은 정수 × scale 그대로).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function

EPS = 1e-11
QMAX = 127
QUANTIZE = True  # False: 양자화 없이 Hadamard 변환만 (train/check_model.py 의 변환 정확성 확인용)


# ---------------------------------------------------------------- Hadamard (HOT utils/hadamard_matmul.py)
def hadamard(n):
    """Sylvester 구성 Hadamard 행렬 (n = 2^k, 원소 ±1)."""
    lg2 = int(math.log(n, 2)) if n >= 1 else 0
    if 2 ** lg2 != n:
        raise ValueError("n must be int and power of 2.")
    H = np.array([[1]], dtype=np.int64)
    for _ in range(lg2):
        H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))
    return H


def block_size(n):
    """n 의 약수 중 가장 큰 2의 거듭제곱, 최대 16 (HOT biggest_power2_factor_max16)."""
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
    """dim 축을 크기 b 블록으로 나눠 블록 대각 Hadamard 를 곱함. (반환, b). 두 번 걸면 b 배.
    메모리 위에서 그대로 (앞, n/b, b, 뒤) 로 보고 b 축에 H 를 곱하므로 결과 텐서 1개만 새로 잡힌다."""
    b = block_size(x.shape[dim])
    if b == 1:
        return x, 1
    v = x.contiguous().view(math.prod(x.shape[:dim]), x.shape[dim] // b, b, math.prod(x.shape[dim + 1:]))
    return torch.matmul(_H_of(b, x.device), v).view(x.shape), b


# ---------------------------------------------------------------- INT8 양자화 (HOT int_quantizer)
def _absmax(x, dim):
    """max|x| (dim=None: 텐서 전체, 아니면 dim 축 원소마다). |x| 텐서를 따로 만들지 않음."""
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
    """대칭 INT8. dim=None 이면 텐서 전체 scale, 아니면 dim 축 원소마다 scale. (int8 텐서, FP32 scale)
    inplace=True 면 x (임시 텐서) 를 계산 공간으로 재사용한다."""
    if not QUANTIZE:
        return x, torch.ones(() if dim is None else x.shape[dim], device=x.device)
    s = _absmax(x, dim) / QMAX
    y = x.div_(_bshape(s, x, dim)) if inplace else x / _bshape(s, x, dim)
    return y.round_().clamp_(-QMAX, QMAX).to(torch.int8), s


def dequant(q, s, dim=None):
    return q.to(torch.float32, copy=True).mul_(_bshape(s, q, dim))


def qdq(x, dim=None, inplace=False):
    """양자화 → 복원 (값 = INT8 정수 × scale). 정수 텐서를 따로 만들지 않고 같은 값을 FP32 로 바로 만든다."""
    if not QUANTIZE:
        return x
    s = _absmax(x, dim) / QMAX
    sb = _bshape(s, x, dim)
    y = x.div_(sb) if inplace else x / sb
    return y.round_().clamp_(-QMAX, QMAX).mul_(sb)


# ---------------------------------------------------------------- 1bit 마스크
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


# ---------------------------------------------------------------- autograd 함수
# 임시 텐서는 쓰고 바로 지운다 (inplace 양자화, del). 값은 위 설명과 같다.
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
        # gx = γ/σ · (gy − Σgy/n − x̂ · Σ(gy·x̂)/n)
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

