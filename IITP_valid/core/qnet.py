"""Quantization-aware ResNet18 (CIFAR stem) shared by training and inference.

Modules match integer-deployment units, so their fx outputs are the stored activations.
Weights: per-channel symmetric [-7, 7]; activations: [0, 15] after ReLU, else [-8, 7].
"""
import math

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F

from . import hot, nipq

TAG = ["other"]  # current op kind, used by memory_calc to group saved tensors


def _tag(name):
    TAG[0] = name


def _hot(m):
    return m.hot and m.training and torch.is_grad_enabled()


def _grad_scale(x, scale):
    return (x - x * scale).detach() + x * scale


def _round_pass(x):
    return (x.round() - x).detach() + x


class LSQ(nn.Module):
    """LSQ quantizer with NIPQ noise during training; eval always rounds."""

    noise = False  # True: NIPQ noise, False: STE rounding

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
            return nipq.SteFn.apply(x, s.detach(), self.qn, self.qp)  # step size frozen
        return _round_pass((x / s).clamp(-self.qn, self.qp)) * s

    @torch.no_grad()
    def int_values(self, x):
        s = self.s.abs().clamp_min(1e-8).view(self._shape(x))
        return (x / s).round().clamp(-self.qn, self.qp)

    @torch.no_grad()
    def init_mse(self, x, n_grid=80):
        """Initialize step size by MSE grid search."""
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


class QInput(nn.Module):

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
        self.wq = None
        self.oq = None
        self.hot = False

    @property
    def out_bits(self):
        return self.oq.bits if self.oq is not None else None

    @property
    def w_bits(self):
        return self.wq.bits if self.wq is not None else None

    @torch.no_grad()
    def fold_bn(self):
        """Fold BN into conv, keeping integer weights and scaling per-channel step size by |gamma/sigma|."""
        if self.bn is None:
            return
        bn, c = self.bn, self.conv
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        shp = (-1, 1, 1, 1)
        if self.wq is not None:
            assert self.wq.narrow, "BN fold keeps integer weights only for narrow (symmetric) range"
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

    def __init__(self):
        super().__init__()
        self.oq = None
        self.hot = False

    def forward(self, x):
        _tag("ReLU")
        y = hot.ReLUFn.apply(x) if _hot(self) else F.relu(x)
        return self.oq(y) if self.oq is not None else y


class QGap(nn.Module):

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
    """ResNet18 with CIFAR stem; num_classes=None returns the final feature map."""

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


def attach_quantizers_uniform(model, bits, w_bits=None):
    """Attach quantizers to every layer, including first and last."""
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
    """Convert to deploy form: fold BN and snap classifier weights to the integer grid."""
    model.fold_bn()
    for q in quantizers(model):
        q.s.abs_().clamp_min_(1e-8)
    for m in model.modules():
        if isinstance(m, QLinear) and m.wq is not None:
            w = m.fc.weight
            w.copy_(m.wq.int_values(w) * m.wq.s.abs().clamp_min(1e-8).view(-1, 1))
    return model


def set_hot(model, on=True):
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
    """MSE-initialize quantizers layer by layer on a calibration batch."""
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


class _Tracer(torch.fx.Tracer):
    def is_leaf_module(self, m, qualname):
        return isinstance(m, (FusedConv, QAdd, QInput, QGap, QLinear)) or super().is_leaf_module(m, qualname)


def trace(model):
    return torch.fx.GraphModule(model, _Tracer().trace(model))


def weight_bytes(model):
    """Deployed weight memory: packed weights + scales + biases + activation qparams."""
    total = 0
    for m in model.modules():
        if isinstance(m, FusedConv):
            c = m.conv
            wb = m.w_bits
            total += math.ceil(c.weight.numel() * (wb if wb else 32) / 8)
            total += c.out_channels * 4  # bias
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
    """Peak live activation bytes from fx liveness, at each tensor's out_bits."""
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
