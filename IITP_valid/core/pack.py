"""Packed W4A4 file format, records in module order:
  weight layer: bit-packed ints (little-endian bit order) + per-channel scale (fp32) + bias (fp32)
  activation quantizer: scale (fp32) + zero point (int32, always 0)
"""
import os

import numpy as np
import torch

from .qnet import FusedConv, QAdd, QGap, QInput, QLinear


def quant_modules(m):
    for mod in m.modules():
        if isinstance(mod, (FusedConv, QLinear)):
            yield "w", mod
        if isinstance(mod, (FusedConv, QAdd, QInput, QGap)) and getattr(mod, "oq", None) is not None:
            yield "a", mod


def pack_bits(q, bits):
    u = (q.astype(np.int16) & (2 ** bits - 1)).astype(np.uint8)
    b = (u[:, None] >> np.arange(bits, dtype=np.uint8)) & 1
    return np.packbits(b.reshape(-1), bitorder="little")


def unpack_bits(buf, n, bits):
    b = np.unpackbits(buf, count=n * bits, bitorder="little").reshape(n, bits).astype(np.int16)
    q = (b << np.arange(bits, dtype=np.int16)).sum(1).astype(np.int16)
    q[q >= 2 ** (bits - 1)] -= 2 ** bits
    return q


@torch.no_grad()
def export_packed(m, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        for kind, mod in quant_modules(m):
            if kind == "w":
                lin = mod.fc if isinstance(mod, QLinear) else mod.conv
                q = mod.wq.int_values(lin.weight).flatten().to(torch.int16).cpu().numpy()
                f.write(pack_bits(q, mod.wq.bits).tobytes())
                f.write(mod.wq.s.abs().float().cpu().numpy().tobytes())
                f.write(lin.bias.float().cpu().numpy().tobytes())
            else:
                f.write(np.array([mod.oq.s.abs().item()], np.float32).tobytes())
                f.write(np.array([0], np.int32).tobytes())
    return os.path.getsize(path)


@torch.no_grad()
def import_packed(m, path):
    buf, off = open(path, "rb").read(), 0

    def take(dtype, n):
        nonlocal off
        arr = np.frombuffer(buf, dtype, n, off)
        off += arr.nbytes
        return arr

    for kind, mod in quant_modules(m):
        if kind == "w":
            lin = mod.fc if isinstance(mod, QLinear) else mod.conv
            n, cout, bits = lin.weight.numel(), lin.weight.size(0), mod.wq.bits
            q = unpack_bits(take(np.uint8, (n * bits + 7) // 8), n, bits)
            scale = torch.from_numpy(take(np.float32, cout).copy())
            bias = torch.from_numpy(take(np.float32, cout).copy())
            dev = lin.weight.device
            shp = (-1,) + (1,) * (lin.weight.dim() - 1)
            lin.weight.copy_((torch.from_numpy(q.astype(np.float32)).view_as(lin.weight.cpu()) * scale.view(shp)).to(dev))
            lin.bias.copy_(bias.to(dev))
            mod.wq.s.copy_(scale.to(dev))
        else:
            mod.oq.s.fill_(float(take(np.float32, 1)[0]))
            take(np.int32, 1)
    assert off == len(buf), "file size mismatch"
    return m
