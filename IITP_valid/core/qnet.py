"""양자화 지원 ResNet18 (torchvision ResNet18 과 동일 구조, CIFAR 32×32 stem) — 학습(Step 1)과 추론(Step 2)이 같이 쓰는 모델 정의.

배포(정수 추론) 단위와 똑같이 모듈을 구성한다:
  QInput    = 입력 이미지 양자화
  FusedConv = conv(+BN, 배포 때는 BN 을 conv 에 합침) + ReLU + 출력 양자화 → 정수 커널 1개, 출력은 out_bits 로 저장
  QAdd      = residual add (+ReLU) + 출력 양자화
  QGap/QLinear = global average pool + 출력 양자화 / 분류기 (가중치 양자화)
  QReLU     = ReLU + 출력 양자화 (SimCLR projection head 에만 씀, 배포 모델에는 없음)
따라서 fx 그래프에서 이 모듈들의 출력 = 메모리에 실제로 저장되는 activation 이다.

양자화기 (LSQ 클래스)
 - 학습 가능한 step size (Esser et al., "LSQ", ICLR'20), MSE 최소화 초기화
 - 가중치: 출력 채널별 step size, 대칭 범위 [-7, 7] (4bit). 대칭이라 BN 을 합칠 때 정수 값이 그대로 유지된다
 - activation: ReLU 뒤는 [0, 15], 그 외 [-8, 7] (4bit)
 - 학습 중: LSQ.noise=True 면 NIPQ 의사양자화 노이즈, False 면 실제 반올림(STE) — nipq.py
 - 평가(eval) 시에는 항상 실제 반올림 → 정수 연산과 같은 값
 - BN folding, 정수 전용 추론: Jacob et al., CVPR'18
학습 backward (hot=True, 학습 중에만): conv·linear·BN·ReLU 의 backward 를 HOT INT8 로 (hot.py)
"""
import math

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F

from . import hot, nipq

TAG = ["other"]  # 지금 실행 중인 연산 종류 (memory_calc 가 backward 저장 텐서를 층 종류별로 나눌 때 씀)


def _tag(name):
    TAG[0] = name


def _hot(m):
    return m.hot and m.training and torch.is_grad_enabled()


# ----------------------------------------------------------------------------
# LSQ / NIPQ quantizer
# ----------------------------------------------------------------------------
def _grad_scale(x, scale):
    return (x - x * scale).detach() + x * scale


def _round_pass(x):
    return (x.round() - x).detach() + x


class LSQ(nn.Module):
    """LSQ step size + NIPQ 학습. per_channel=True 이면 dim0(출력 채널)별 step size. narrow=True 면 대칭 [-(2^(b-1)-1), 2^(b-1)-1]."""

    noise = False  # True: NIPQ 의사양자화(학습 중 반올림 대신 균일 노이즈). 평가(eval) 시에는 항상 실제 반올림.

    def __init__(self, bits, signed, per_channel=False, channels=1, narrow=False, kind="a"):
        super().__init__()
        self.bits, self.signed, self.per_channel, self.narrow, self.kind = bits, signed, per_channel, narrow, kind
        self.s = nn.Parameter(torch.ones(channels if per_channel else 1))
        self.enabled = True

    @property
    def qn(self):
        return (2 ** (self.bits - 1) - int(self.narrow)) if self.signed else 0

    @property
    def qp(self):
        return 2 ** (self.bits - 1) - 1 if self.signed else 2 ** self.bits - 1

    def _shape(self, x):
        return (-1,) + (1,) * (x.dim() - 1) if self.per_channel else (1,) * x.dim()

    def forward(self, x):
        if not self.enabled or self.bits is None:
            return x
        _tag("NIPQ(weight)" if self.kind == "w" else "NIPQ(activation)")
        g = 1.0 / math.sqrt(x[0].numel() * self.qp)
        s = _grad_scale(self.s, g).abs().clamp_min(1e-8).view(self._shape(x))
        if self.training and LSQ.noise:
            seed = nipq.new_seed()
            if nipq.EFFICIENT:
                return nipq.NoiseFn.apply(x, s, self.qn, self.qp, seed)
            return nipq.noise_naive(x, s, self.qn, self.qp, seed)
        if self.training and torch.is_grad_enabled() and nipq.EFFICIENT:
            return nipq.SteFn.apply(x, s.detach(), self.qn, self.qp)  # STE 미세조정: step size 고정
        return _round_pass((x / s).clamp(-self.qn, self.qp)) * s

    @torch.no_grad()
    def int_values(self, x):
        """실제 정수 값 (검증·배포용): round(clamp(x/s))."""
        s = self.s.abs().clamp_min(1e-8).view(self._shape(x))
        return (x / s).round().clamp(-self.qn, self.qp)

    @torch.no_grad()
    def init_mse(self, x, n_grid=80):
        """MSE 최소화 step size 탐색 초기화 (LSQ+/PTQ 표준 방식)."""
        if self.per_channel:
            xf = x.reshape(x.size(0), -1)
            amax = xf.abs().amax(1).clamp_min(1e-8)
        else:
            xf = x.reshape(1, -1)
            if xf.size(1) > 2_000_000:
                xf = xf[:, torch.randperm(xf.size(1), device=x.device)[:2_000_000]]
            amax = xf.abs().amax(1).clamp_min(1e-8) if self.signed else xf.clamp_min(0).amax(1).clamp_min(1e-8)
        best_err = torch.full_like(amax, float("inf"))
        best_s = amax / self.qp
        for r in torch.linspace(0.05, 1.0, n_grid, device=x.device):
            s = (amax * r / self.qp).unsqueeze(1)
            q = (xf / s).round().clamp(-self.qn, self.qp) * s
            err = (q - xf).pow(2).mean(1)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_s = torch.where(better, s.squeeze(1), best_s)
        self.s.data.copy_(best_s if self.per_channel else best_s.view(1))


# ----------------------------------------------------------------------------
# Deploy-unit modules
# ----------------------------------------------------------------------------
class QInput(nn.Module):
    """입력 이미지 양자화 (정수 추론 시 첫 텐서)."""

    def __init__(self):
        super().__init__()
        self.oq = None

    @property
    def out_bits(self):
        return self.oq.bits if self.oq is not None else None

    def forward(self, x):
        return self.oq(x) if self.oq is not None else x


class FusedConv(nn.Module):
    def __init__(self, cin, cout, k, stride, act):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = act
        self.wq = None  # weight quantizer
        self.oq = None  # output(activation) quantizer
        self.hot = False

    @property
    def out_bits(self):
        return self.oq.bits if self.oq is not None else None

    @property
    def w_bits(self):
        return self.wq.bits if self.wq is not None else None

    @torch.no_grad()
    def fold_bn(self):
        """BN(running 통계)을 conv 에 합침. 가중치 양자화기가 있으면 정수 값은 그대로 두고 채널별 step size 에 |γ/σ| 를 곱한다
        (부호는 정수 쪽으로, 대칭 범위라 범위를 넘지 않음) → 합치기 전 eval 출력과 같은 값."""
        if self.bn is None:
            return
        bn, c = self.bn, self.conv
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        shp = (-1, 1, 1, 1)
        if self.wq is not None:
            assert self.wq.narrow, "대칭 범위 가중치만 정수를 유지한 채 합칠 수 있음"
            s = self.wq.s.abs().clamp_min(1e-8) * scale.abs()
            w = self.wq.int_values(c.weight) * scale.sign().view(shp) * s.view(shp)
            self.wq.s.copy_(s)
        else:
            w = c.weight * scale.view(shp)
        conv = nn.Conv2d(c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding, bias=True).to(w.device, w.dtype)
        conv.weight.copy_(w)
        conv.bias.copy_(bn.bias - bn.running_mean * scale)
        self.conv, self.bn = conv, None

    def forward(self, x):
        c = self.conv
        w = self.wq(c.weight) if self.wq is not None else c.weight
        _tag("Conv2d")
        if _hot(self):
            y = hot.ConvFn.apply(x, w, c.bias, c.stride, c.padding, c.dilation)
        else:
            y = F.conv2d(x, w, c.bias, c.stride, c.padding)
        if self.bn is not None:
            _tag("BatchNorm2d")
            if _hot(self):
                bn = self.bn
                bn.num_batches_tracked.add_(1)
                y = hot.BNFn.apply(y, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.momentum, bn.eps)
            else:
                y = self.bn(y)
        if self.act:
            _tag("ReLU")
            y = hot.ReLUFn.apply(y) if _hot(self) else F.relu(y)
        return self.oq(y) if self.oq is not None else y


class QAdd(nn.Module):
    """residual add (+ReLU) + 출력 양자화."""

    def __init__(self, act=False):
        super().__init__()
        self.act = act
        self.oq = None
        self.hot = False

    @property
    def out_bits(self):
        return self.oq.bits if self.oq is not None else None

    def forward(self, a, b):
        _tag("Add")
        y = a + b
        if self.act:
            _tag("ReLU")
            y = hot.ReLUFn.apply(y) if _hot(self) else F.relu(y)
        return self.oq(y) if self.oq is not None else y


class QReLU(nn.Module):
    """ReLU + 출력 양자화 (SimCLR projection head)."""

    def __init__(self):
        super().__init__()
        self.oq = None
        self.hot = False

    def forward(self, x):
        _tag("ReLU")
        y = hot.ReLUFn.apply(x) if _hot(self) else F.relu(x)
        return self.oq(y) if self.oq is not None else y


class QGap(nn.Module):
    """global average pool + 출력 양자화."""

    def __init__(self):
        super().__init__()
        self.oq = None

    @property
    def out_bits(self):
        return self.oq.bits if self.oq is not None else None

    def forward(self, x):
        _tag("GAP")
        y = x.mean((2, 3))
        return self.oq(y) if self.oq is not None else y


class QLinear(nn.Module):
    """fc (가중치 양자화). 출력은 FP32."""

    def __init__(self, cin, cout):
        super().__init__()
        self.fc = nn.Linear(cin, cout)
        self.wq = None
        self.out_bits = None
        self.hot = False

    @property
    def w_bits(self):
        return self.wq.bits if self.wq is not None else None

    def forward(self, x):
        w = self.wq(self.fc.weight) if self.wq is not None else self.fc.weight
        _tag("Linear")
        if _hot(self):
            return hot.LinearFn.apply(x, w, self.fc.bias)
        return F.linear(x, w, self.fc.bias)


class Head(nn.Module):
    def __init__(self, cin, num_classes):
        super().__init__()
        self.gap, self.fc = QGap(), QLinear(cin, num_classes)

    def forward(self, x):
        return self.fc(self.gap(x))


class BasicBlock(nn.Module):
    """torchvision ResNet BasicBlock 과 동일 (conv-bn-relu, conv-bn, (+downsample), add, relu)."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = FusedConv(cin, cout, 3, stride, act=True)
        self.c2 = FusedConv(cout, cout, 3, 1, act=False)
        self.ds = FusedConv(cin, cout, 1, stride, act=False) if stride != 1 or cin != cout else None
        self.add = QAdd(act=True)

    def forward(self, x):
        y = self.c2(self.c1(x))
        return self.add(y, self.ds(x) if self.ds is not None else x)


class ResNet18(nn.Module):
    """torchvision ResNet18 과 동일 구조, CIFAR(32×32) stem (conv1 3×3/s1, maxpool 없음).
    num_classes=None 이면 분류기 없이 마지막 특징 맵 (N, 512, 4, 4) 을 낸다 (SimCLR 백본)."""

    def __init__(self, num_classes=None):
        super().__init__()
        self.inp = QInput()
        self.stem = FusedConv(3, 64, 3, 1, act=True)
        blocks, cin = [], 64
        for cout, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            blocks += [BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1)]
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.feat_dim = 512
        self.head = Head(512, num_classes) if num_classes else None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.blocks(self.stem(self.inp(x)))
        return self.head(x) if self.head is not None else x

    def fused_convs(self):
        return [m for m in self.modules() if isinstance(m, FusedConv)]

    def fold_bn(self):
        for m in self.fused_convs():
            m.fold_bn()
        return self


def build_model(num_classes=10):
    return ResNet18(num_classes=num_classes)


# ----------------------------------------------------------------------------
# Quantization / HOT config
# ----------------------------------------------------------------------------
def attach_quantizers_uniform(model, bits, w_bits=None):
    """양자화기 부착: 모든 층(입력·첫 층·마지막 층·분류기 포함)의 activation 은 bits, 가중치는 w_bits (없으면 bits).
    예외 층 없음. W4A4 = (bits=4)."""
    dev = next(model.parameters()).device
    w_bits = w_bits or bits
    for m in model.modules():
        if isinstance(m, FusedConv):
            m.wq = LSQ(w_bits, True, True, m.conv.out_channels, narrow=True, kind="w").to(dev)
            m.oq = LSQ(bits, signed=not m.act).to(dev)
        elif isinstance(m, QAdd):
            m.oq = LSQ(bits, signed=not m.act).to(dev)
        elif isinstance(m, QInput):
            m.oq = LSQ(bits, signed=True).to(dev)
        elif isinstance(m, (QGap, QReLU)):
            m.oq = LSQ(bits, signed=False).to(dev)
        elif isinstance(m, QLinear):
            m.wq = LSQ(w_bits, True, True, m.fc.out_features, narrow=True, kind="w").to(dev)
    return model


@torch.no_grad()
def to_deploy(model):
    """학습 모델 → 배포 형태: BN 을 conv 에 합치고 (정수 가중치 그대로), 분류기 가중치도 정수 격자 값(정수 × scale)으로 고정.
    step size 는 |s| 로 저장 (학습 중에는 부호와 무관하게 |s| 를 씀)."""
    model.fold_bn()
    for q in quantizers(model):
        q.s.abs_().clamp_min_(1e-8)
    for m in model.modules():
        if isinstance(m, QLinear) and m.wq is not None:
            w = m.fc.weight
            w.copy_(m.wq.int_values(w) * m.wq.s.abs().clamp_min(1e-8).view(-1, 1))
    return model


def set_hot(model, on=True):
    """학습 중 backward 를 HOT INT8 로 (FusedConv·QAdd·QReLU·QLinear)."""
    for m in model.modules():
        if hasattr(m, "hot"):
            m.hot = on
    return model


def set_quant(model, on=True):
    for m in model.modules():
        if isinstance(m, LSQ):
            m.enabled = on
    return model


def quantizers(model):
    return [m for m in model.modules() if isinstance(m, LSQ)]


@torch.no_grad()
def calibrate(model, x):
    """가중치 quantizer 는 가중치로, activation quantizer 는 보정 배치 입력으로 MSE 초기화 (라벨 불필요).
    앞 레이어부터 순차적으로 켜면서 초기화 → 누적 양자화 오차 반영."""
    was_training = model.training
    model.eval()
    for m in model.modules():
        if isinstance(m, FusedConv) and m.wq is not None:
            m.wq.enabled = True
            m.wq.init_mse(m.conv.weight)
        if isinstance(m, QLinear) and m.wq is not None:
            m.wq.enabled = True
            m.wq.init_mse(m.fc.weight)
    act_qs = [q for q in quantizers(model) if q.kind == "a"]
    for q in act_qs:
        q.enabled = False
    for q in act_qs:
        cache = {}
        h = q.register_forward_pre_hook(lambda mod, inp: cache.__setitem__("x", inp[0].detach()))
        model(x)
        h.remove()
        q.init_mse(cache["x"])
        q.enabled = True
    model.train(was_training)
    return model


# ----------------------------------------------------------------------------
# Memory accounting (bit-width aware, 추론)
# ----------------------------------------------------------------------------
class _Tracer(torch.fx.Tracer):
    def is_leaf_module(self, m, qualname):
        return isinstance(m, (FusedConv, QAdd, QInput, QGap, QLinear)) or super().is_leaf_module(m, qualname)


def trace(model):
    return torch.fx.GraphModule(model, _Tracer().trace(model))


def weight_bytes(model):
    """정수 배포 기준 가중치 메모리: 가중치(bit 폭) + per-channel scale(fp32) + bias(int32/fp32) + 활성 qparam."""
    total = 0
    for m in model.modules():
        if isinstance(m, FusedConv):
            c = m.conv
            wb = m.w_bits
            total += math.ceil(c.weight.numel() * (wb if wb else 32) / 8)
            total += c.out_channels * 4  # bias (int32 or fp32)
            if wb:
                total += c.out_channels * 4  # per-channel weight scale
            if m.bn is not None:
                total += 4 * c.out_channels * 4
        if isinstance(m, QLinear):
            f = m.fc
            total += math.ceil(f.weight.numel() * (m.w_bits or 32) / 8) + f.out_features * 4
            if m.w_bits:
                total += f.out_features * 4
        if isinstance(m, (FusedConv, QAdd, QInput, QGap)) and getattr(m, "oq", None) is not None:
            total += 8  # activation scale + zero point
    return total


def peak_activation_bytes(model, x, input_bytes_per_elem=4):
    """fx liveness 기반 피크 activation. 저장 텐서 크기 = numel * out_bits/8 (FP 는 4B).
    입력 이미지는 원소당 input_bytes_per_elem 바이트(기본 FP32 4B)로 센다."""
    from torch.fx.passes.shape_prop import ShapeProp
    gm = trace(model)
    ShapeProp(gm).propagate(x)
    nodes = list(gm.graph.nodes)
    order = {n: i for i, n in enumerate(nodes)}

    def nbytes(n):
        meta = n.meta.get("tensor_meta")
        if meta is None or not hasattr(meta, "shape"):
            return 0
        numel = 1
        for s in meta.shape:
            numel *= int(s)
        if n.op == "placeholder":
            return numel * input_bytes_per_elem
        if n.op == "call_module":
            bits = getattr(gm.get_submodule(n.target), "out_bits", None)
            if bits:
                return math.ceil(numel * bits / 8)
        return numel * 4

    size = {n: nbytes(n) for n in nodes if n.op not in ("output", "get_attr")}
    last = {n: max([order[u] for u in n.users] + [order[n]]) for n in size}
    live, cur, peak, at = set(), 0, 0, None
    for i, n in enumerate(nodes):
        if n in size:
            live.add(n)
            cur += size[n]
        if cur > peak:
            peak, at = cur, n.name
        for b in [b for b in live if last[b] <= i]:
            live.remove(b)
            cur -= size[b]
    return peak, at
