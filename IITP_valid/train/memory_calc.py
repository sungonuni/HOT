"""Step 1 analytic training memory for one SimCLR + ResNet18 step.

Total = weights + gradients + optimizer state (SGD momentum) + activations saved for backward.
Saved activations are collected with saved_tensors_hooks, deduplicated by storage, parameters excluded.

    python memory_calc.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from core import nipq, qnet  # noqa: E402
from core.common import BATCH, MODE_EN, MiB, TRAIN_TARGET, pretrain_ckpt  # noqa: E402
from core.data import load_cifar10, simclr_view, to_input  # noqa: E402
from core.model import build_simclr, nt_xent  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CALC_MODES = ["fp32", "nipq_hot", "nipq_hot_naive"]


def train_model(mode, dev, calib_x=None):
    """Model in training state; loads the pretrain ckpt if present, else calibrates step sizes."""
    m = build_simclr(mode).to(dev)
    ck = pretrain_ckpt(mode.replace("_naive", ""))
    if os.path.exists(ck):
        m.load_state_dict(torch.load(ck, map_location="cpu")["state_dict"])
    elif mode != "fp32" and calib_x is not None:
        qnet.calibrate(m, calib_x)
    nipq.EFFICIENT = mode != "nipq_hot_naive"
    qnet.LSQ.noise = mode != "fp32"
    return m.train()


def saved_activations(model, x):
    """Tensors saved for backward during one forward + loss: [(layer type, dtype, shape, bytes)]."""
    param_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen, recs = set(), []

    def pack(t):
        ptr = t.untyped_storage().data_ptr()
        if ptr not in seen and ptr not in param_storages:
            seen.add(ptr)
            recs.append((qnet.TAG[0], str(t.dtype).replace("torch.", ""), list(t.shape), t.untyped_storage().nbytes()))
        return t

    qnet.TAG[0] = "other"
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = nt_xent(model(x))
    del loss
    return recs


def calc(mode, x, calib_x):
    model = train_model(mode, x.device, calib_x)
    n_par = sum(p.numel() for p in model.parameters())
    n_buf = sum(b.numel() * b.element_size() for b in model.buffers())
    recs = saved_activations(model, x)
    nipq.EFFICIENT, qnet.LSQ.noise = True, False
    by_type, by_dtype = {}, {}
    for typ, dt, _, nb in recs:
        by_type[typ] = by_type.get(typ, 0) + nb
        by_dtype[dt] = by_dtype.get(dt, 0) + nb
    r = {"weight_bytes": n_par * 4 + n_buf, "gradient_bytes": n_par * 4, "optimizer_state_bytes": n_par * 4,
         "activation_bytes": sum(nb for *_, nb in recs), "activation_by_layer_type": by_type,
         "activation_by_dtype": by_dtype, "n_saved_tensors": len(recs), "params": n_par}
    r["total_bytes"] = r["weight_bytes"] + r["gradient_bytes"] + r["optimizer_state_bytes"] + r["activation_bytes"]
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data", default=None)
    p.add_argument("--output-dir", default=RESULTS)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    torch.manual_seed(0)
    dev = torch.device(a.device)
    xtr, *_ = load_cifar10("cpu", a.data)
    xb = xtr[:BATCH].to(dev)
    with torch.no_grad():
        x = torch.cat([simclr_view(xb), simclr_view(xb)])
    out = {"batch": f"{BATCH} images x 2 views = {x.size(0)}x3x32x32", "optimizer": "SGD momentum 0.9"}
    for mode in CALC_MODES:
        out[mode] = calc(mode, x, to_input(xb))
    f = out["fp32"]
    for mode in CALC_MODES[1:]:
        out[mode]["reduction_pct"] = 100 * (1 - out[mode]["total_bytes"] / f["total_bytes"])
        out[mode]["activation_reduction_pct"] = 100 * (1 - out[mode]["activation_bytes"] / f["activation_bytes"])
    out["reduction_pct"] = out["nipq_hot"]["reduction_pct"]
    json.dump(out, open(os.path.join(a.output_dir, "memory_calc.json"), "w"), indent=1)
    row = lambda k: "".join(f"{out[m][k] / MiB:12.1f}" for m in CALC_MODES)  # noqa: E731
    h = out["nipq_hot"]
    print(f"{f'Train memory calc (MiB, batch {x.size(0)})':32}{'FP32':>12}{'NIPQ+HOT':>12}{'naive+HOT':>12}\n"
          f"  weights                         {row('weight_bytes')}\n"
          f"  gradients                       {row('gradient_bytes')}\n"
          f"  optimizer state (momentum)      {row('optimizer_state_bytes')}\n"
          f"  activations (saved for bwd)     {row('activation_bytes')}\n"
          f"  total                           {row('total_bytes')}\n"
          f"  NIPQ+HOT reduction {h['reduction_pct']:.1f}% {'PASS' if h['reduction_pct'] >= TRAIN_TARGET else 'FAIL'}"
          f" | (ref) naive NIPQ+HOT reduction {out['nipq_hot_naive']['reduction_pct']:.1f}%", flush=True)
    for mode in CALC_MODES:
        print(f"  [{MODE_EN[mode]}] activations by layer type (MiB): "
              + ", ".join(f"{k} {v / MiB:.1f}" for k, v in out[mode]["activation_by_layer_type"].items())
              + " | by dtype: " + ", ".join(f"{k} {v / MiB:.1f}" for k, v in out[mode]["activation_by_dtype"].items()))


if __name__ == "__main__":
    main()
