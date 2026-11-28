"""CIFAR-10 loading, GPU augmentation, label subsets, accuracy."""
import math
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from .common import ROOT

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def _from_bin(d):
    def rd(f):
        raw = np.fromfile(os.path.join(d, f), dtype=np.uint8).reshape(-1, 3073)
        return raw[:, 1:].reshape(-1, 3, 32, 32), raw[:, 0].astype(np.int64)
    xs, ys = zip(*[rd(f"data_batch_{i}.bin") for i in range(1, 6)])
    xte, yte = rd("test_batch.bin")
    return np.concatenate(xs), np.concatenate(ys), xte, yte


def _from_py(d):
    def rd(f):
        with open(os.path.join(d, f), "rb") as fh:
            b = pickle.load(fh, encoding="bytes")
        return np.asarray(b[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32), np.asarray(b[b"labels"], dtype=np.int64)
    xs, ys = zip(*[rd(f"data_batch_{i}") for i in range(1, 6)])
    xte, yte = rd("test_batch")
    return np.concatenate(xs), np.concatenate(ys), xte, yte


def load_cifar10(device, root=None):
    """Load CIFAR-10 as uint8 tensors; download via torchvision if not found under root."""
    root = root or os.path.join(ROOT, "data")
    os.makedirs(root, exist_ok=True)
    import fcntl
    with open(os.path.join(root, ".download.lock"), "w") as lk:  # serialize concurrent downloads
        fcntl.flock(lk, fcntl.LOCK_EX)
        for sub, fn in [("cifar-10-batches-bin", _from_bin), ("cifar-10-batches-py", _from_py)]:
            if os.path.isdir(os.path.join(root, sub)):
                arrs = fn(os.path.join(root, sub))
                break
        else:
            import torchvision
            torchvision.datasets.CIFAR10(root, train=False, download=True)
            arrs = _from_py(os.path.join(root, "cifar-10-batches-py"))
    return tuple(torch.from_numpy(a).to(device) for a in arrs)


def label_subset(y, frac, seed=0, classes=10):
    """Class-balanced random subset of train indices (seeded)."""
    if frac >= 1:
        return torch.arange(y.numel(), device=y.device)
    g = torch.Generator().manual_seed(seed)
    yc = y.cpu()
    idx = []
    for c in range(classes):
        ic = (yc == c).nonzero().view(-1)
        idx.append(ic[torch.randperm(ic.numel(), generator=g)[:round(ic.numel() * frac)]])
    return torch.cat(idx).sort().values.to(y.device)


def normalize(x01):
    return (x01 - MEAN.to(x01.device)) / STD.to(x01.device)


def to_input(x_uint8):
    return normalize(x_uint8.float() / 255.0)


def augment(x_uint8):
    """Pad-4 random crop + horizontal flip."""
    n, dev = x_uint8.size(0), x_uint8.device
    x = F.pad(to_input(x_uint8), (4, 4, 4, 4))
    i = torch.randint(0, 9, (n, 1, 1, 1), device=dev)
    j = torch.randint(0, 9, (n, 1, 1, 1), device=dev)
    ar = torch.arange(32, device=dev)
    x = x.gather(2, (i + ar.view(1, 1, 32, 1)).expand(n, 3, 32, 40))
    x = x.gather(3, (j + ar.view(1, 1, 1, 32)).expand(n, 3, 32, 32))
    flip = torch.rand(n, device=dev) < 0.5
    x[flip] = x[flip].flip(3)
    return x


def _gray(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def simclr_view(x_uint8, min_scale=0.2):
    """One SimCLR view: RandomResizedCrop(0.2-1) + flip, ColorJitter(0.4, 0.4, 0.4, 0.1) p=0.8, Grayscale p=0.2."""
    n, dev = x_uint8.size(0), x_uint8.device
    x = x_uint8.float() / 255.0
    u = lambda lo, hi, *s: torch.empty(n, *s, device=dev).uniform_(lo, hi)  # noqa: E731
    # RandomResizedCrop + flip via per-image affine grid
    area = u(min_scale, 1.0)
    logr = u(math.log(3 / 4), math.log(4 / 3))
    w = torch.sqrt(area * torch.exp(logr)).clamp(max=1.0)
    h = torch.sqrt(area / torch.exp(logr)).clamp(max=1.0)
    cx, cy = (torch.rand(n, device=dev) * 2 - 1) * (1 - w), (torch.rand(n, device=dev) * 2 - 1) * (1 - h)
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1], theta[:, 1, 2] = w * flip, cx, h, cy
    grid = F.affine_grid(theta, (n, 3, 32, 32), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    # ColorJitter: brightness, contrast, saturation, hue
    jit = (torch.rand(n, 1, 1, 1, device=dev) < 0.8).float()
    y = x * u(0.6, 1.4, 1, 1, 1)
    y = (y - _gray(y).mean((2, 3), keepdim=True)) * u(0.6, 1.4, 1, 1, 1) + _gray(y).mean((2, 3), keepdim=True)
    y = y.clamp(0, 1)
    g = _gray(y)
    y = ((y - g) * u(0.6, 1.4, 1, 1, 1) + g).clamp(0, 1)
    # hue: rotate the IQ plane in YIQ space
    ang = u(-0.1, 0.1) * 2 * math.pi
    yiq = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], device=dev)
    c = torch.einsum("ij,njhw->nihw", yiq, y)
    cos, sin = torch.cos(ang).view(n, 1, 1), torch.sin(ang).view(n, 1, 1)
    i2, q2 = c[:, 1] * cos - c[:, 2] * sin, c[:, 1] * sin + c[:, 2] * cos
    y = torch.einsum("ij,njhw->nihw", torch.linalg.inv(yiq), torch.stack([c[:, 0], i2, q2], 1)).clamp(0, 1)
    x = jit * y + (1 - jit) * x
    gr = (torch.rand(n, 1, 1, 1, device=dev) < 0.2).float()
    x = gr * _gray(x).expand(-1, 3, -1, -1) + (1 - gr) * x
    return normalize(x)


@torch.no_grad()
def accuracy(model, x_uint8, y, bs=1000):
    model.eval()
    correct = 0
    for k in range(0, x_uint8.size(0), bs):
        correct += (model(to_input(x_uint8[k:k + bs])).argmax(1) == y[k:k + bs]).sum().item()
    return 100.0 * correct / x_uint8.size(0)


def cosine_lr(step, total, base, warmup=0):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
