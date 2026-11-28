"""Runs the packed W4A4 .bin directly: weights and activations stay 4-bit packed on GPU and are
dequantized per channel chunk (chunk=None: whole layer) only while computing.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp

import common  # noqa: F401  (adds core to sys.path)
from core.pack import quant_modules
from core.qnet import FusedConv, QAdd, QGap, QInput, QLinear, trace


def unpack(p, start, end, signed, bits=4):
    """Unpack 4-bit elements [start, end) of p (N, L) uint8 -> int8/uint8 (N, end-start)."""
    assert bits == 4
    b0 = start // 2
    seg = p[:, b0:(end + 1) // 2]
    v = torch.empty(p.size(0), 2 * seg.size(1), dtype=torch.int8 if signed else torch.uint8, device=p.device)
    v[:, 0::2] = seg & 15
    v[:, 1::2] = seg >> 4
    if signed:
        v.bitwise_xor_(8).sub_(8)  # 4-bit two's complement -> signed
    return v[:, start - 2 * b0:start - 2 * b0 + end - start]


def pack_into(out, start, q):
    """Pack 4-bit integer values q (N, n) into out (N, L) uint8 at element offset start (even)."""
    assert start % 2 == 0
    nib = q.to(torch.int8).bitwise_and_(15).view(torch.uint8)
    if nib.size(1) % 2:
        nib = F.pad(nib, (0, 1))
    out[:, start // 2:start // 2 + nib.size(1) // 2] = nib[:, 0::2] | (nib[:, 1::2] << 4)


class QT:
    """Packed 4-bit activation."""
    __slots__ = ("p", "s", "signed", "shape")

    def __init__(self, p, s, signed, shape):
        self.p, self.s, self.signed, self.shape = p, s, signed, shape


class PackedRuntime:
    def __init__(self, qmodel, packed_file, device="cuda", chunk=16, dtype=torch.float32, res=32):
        qmodel = qmodel.cpu().eval()
        self.gm = trace(qmodel)
        ShapeProp(self.gm).propagate(torch.randn(1, 3, res, res))
        self.chunk, self.dt, self.dev = chunk, dtype, device
        raw = open(packed_file, "rb").read()
        self.blob = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
        self.weight_bytes = self.blob.numel()
        self.w, self.a, off = {}, {}, 0
        for kind, mod in quant_modules(qmodel):  # same order as pack.export_packed
            if kind == "w":
                lin = mod.fc if isinstance(mod, QLinear) else mod.conv
                n, cout, bits = lin.weight.numel(), lin.weight.size(0), mod.wq.bits
                nb = (n * bits + 7) // 8
                wp = self.blob[off:off + nb]
                off += nb
                ws, b = (self._f32(off + 4 * cout * i, cout) for i in range(2))
                off += 8 * cout
                self.w[id(mod)] = (wp, ws, b, tuple(lin.weight.shape), bits)
            else:
                self.a[id(mod)] = (float(np.frombuffer(raw, np.float32, 1, off)[0]), mod.oq.qn, mod.oq.qp)
                off += 8
        assert off == len(raw)
        nodes = list(self.gm.graph.nodes)
        order = {n: i for i, n in enumerate(nodes)}
        self.free_after = {i: [] for i in range(len(nodes))}
        for n in nodes:
            if n.users:
                self.free_after[max(order[u] for u in n.users)].append(n)
        self.nodes = nodes

    def _f32(self, off, n):
        if off % 4 == 0:
            return self.blob[off:off + 4 * n].view(torch.float32)
        return self.blob[off:off + 4 * n].clone().view(torch.float32)  # unaligned: copy

    def _deq(self, t, c0, c1):
        C, H, W = t.shape
        v = unpack(t.p, c0 * H * W, c1 * H * W, t.signed)
        return v.to(self.dt).mul_(t.s).view(-1, c1 - c0, H, W)

    def _quant_out(self, mod, shape, n):
        s, qn, qp = self.a[id(mod)]
        out = torch.empty(n, math.ceil(math.prod(shape) / 2), dtype=torch.uint8, device=self.dev)
        return QT(out, s, qn > 0, shape), -qn, qp

    def _write(self, qt, lo, hi, c0, y):
        H, W = qt.shape[1:] if len(qt.shape) == 3 else (1, 1)
        y = y.float() if y.dtype != torch.float32 else y
        q = y.div_(qt.s).round_().clamp_(lo, hi)
        pack_into(qt.p, c0 * H * W, q.reshape(q.size(0), -1))

    @torch.inference_mode()
    def __call__(self, x):
        env = {}
        N = x.size(0)
        for i, node in enumerate(self.nodes):
            if node.op == "placeholder":
                env[node] = x
            elif node.op == "output":
                return env[node.args[0]]
            else:
                m = self.gm.get_submodule(node.target)
                ins = [env[a] for a in node.args]
                shape = tuple(int(s) for s in node.meta["tensor_meta"].shape[1:])
                env[node] = self._run(m, ins, shape, N)
            for dead in self.free_after[i]:
                env.pop(dead, None)

    def _chunks(self, C):
        k = C if self.chunk is None else self.chunk
        return [(c0, min(c0 + k, C)) for c0 in range(0, C, k)]

    def _run(self, m, ins, shape, N):
        if isinstance(m, QInput):
            out, lo, hi = self._quant_out(m, shape, N)
            for c0, c1 in self._chunks(shape[0]):
                self._write(out, lo, hi, c0, ins[0][:, c0:c1].float().clone())
            return out
        if isinstance(m, FusedConv):
            t = ins[0]
            conv = m.conv
            wp, ws, b, wshape, wbits = self.w[id(m)]
            per = math.prod(wshape[1:])
            out, lo, hi = self._quant_out(m, shape, N)
            for c0, c1 in self._chunks(wshape[0]):
                wi = unpack(wp[None], c0 * per, c1 * per, True, wbits)[0].view(c1 - c0, *wshape[1:])
                sw = ws[c0:c1].view(-1, 1, 1, 1).to(self.dt)
                acc = None
                for i0, i1 in self._chunks(wshape[1]):  # accumulate partial sums in fp32
                    w = wi[:, i0:i1].to(self.dt).mul_(sw)
                    part = F.conv2d(self._deq(t, i0, i1), w, None, conv.stride, conv.padding)
                    acc = part.float() if acc is None else acc.add_(part)
                    del part, w
                del wi
                acc.add_(b[c0:c1].view(1, -1, 1, 1))
                if m.act:
                    acc = acc.relu_()
                self._write(out, lo, hi, c0, acc)
                del acc
            return out
        if isinstance(m, QAdd):
            a, bb = ins
            out, lo, hi = self._quant_out(m, shape, N)
            for c0, c1 in self._chunks(shape[0]):
                y = self._deq(a, c0, c1).add_(self._deq(bb, c0, c1))
                if m.act:
                    y = y.relu_()
                self._write(out, lo, hi, c0, y)
            return out
        if isinstance(m, QGap):
            t = ins[0]
            out, lo, hi = self._quant_out(m, shape, N)
            for c0, c1 in self._chunks(t.shape[0]):
                self._write(out, lo, hi, c0, self._deq(t, c0, c1).float().mean((2, 3)))
            return out
        if isinstance(m, QLinear):
            t = ins[0]
            x = (unpack(t.p, 0, t.shape[0], t.signed).to(self.dt) * t.s)
            wp, ws, b, wshape, wbits = self.w[id(m)]
            w = unpack(wp[None], 0, math.prod(wshape), True, wbits)[0].to(self.dt).view(wshape) * ws[:, None].to(self.dt)
            return F.linear(x, w).float() + b
        raise TypeError(type(m))
