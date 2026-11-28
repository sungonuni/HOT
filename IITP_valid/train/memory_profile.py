"""Step 1 measured GPU training memory (one step) and memory timeline plot.

Each mode runs in its own process; --combine stacks the plots on a shared Y axis.
peak_training_tensors excludes the transient cuDNN conv workspace; peak_allocated includes it.

    python memory_profile.py --mode fp32|nipq_hot|nipq_hot_naive
    python memory_profile.py --combine
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")
import tempfile  # noqa: E402

_ntf = tempfile.NamedTemporaryFile
# Workaround: export_memory_timeline(html) opens a temp PNG with "wb" and then reads it back.
tempfile.NamedTemporaryFile = lambda mode="w+b", *a, **k: _ntf("w+b" if mode == "wb" else mode, *a, **k)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from core.common import BATCH, MODE_EN, NAME, MiB  # noqa: E402
from core.data import load_cifar10, to_input  # noqa: E402
from core.model import make_optimizer, train_step  # noqa: E402
from memory_calc import CALC_MODES, RESULTS, train_model  # noqa: E402

YLIM_MIB = 7000


def replay_peak(snapshot, device_index):
    """Replay allocations: (peak, peak without cuDNN workspace, peak cuDNN workspace)."""
    ws, cur_all, cur_model, cur_ws, pk_all, pk_model, pk_ws = set(), 0, 0, 0, 0, 0, 0
    for e in snapshot["device_traces"][device_index]:
        if e["action"] == "alloc":
            # cuDNN workspace is allocated only inside run_conv_plan
            is_ws = any("run_conv_plan" in f["name"] for f in e["frames"])
            cur_all += e["size"]
            if is_ws:
                ws.add(e["addr"])
                cur_ws += e["size"]
            else:
                cur_model += e["size"]
        elif e["action"] == "free_completed":
            cur_all -= e["size"]
            if e["addr"] in ws:
                ws.discard(e["addr"])
                cur_ws -= e["size"]
            else:
                cur_model -= e["size"]
        pk_all, pk_model, pk_ws = max(pk_all, cur_all), max(pk_model, cur_model), max(pk_ws, cur_ws)
    return pk_all, pk_model, pk_ws


def capture(a):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    torch.manual_seed(0)
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    xtr, *_ = load_cifar10("cpu", a.data)
    xb = xtr[:BATCH].to(dev)
    model = train_model(a.mode, dev, to_input(xb))
    opt = make_optimizer(model, lr=1e-3)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    for _ in range(3):  # warmup
        train_step(model, opt, xb)
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    before = torch.cuda.memory_allocated(dev)
    torch.cuda.memory._record_memory_history(max_entries=2_000_000, stacks="all")
    train_step(model, opt, xb)
    torch.cuda.synchronize(dev)
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    peak = torch.cuda.max_memory_allocated(dev)
    pk_all, pk_model, pk_ws = replay_peak(snap, idx)
    n_par = sum(p.numel() for p in model.parameters())
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                 profile_memory=True, with_stack=True) as prof:
        train_step(model, opt, xb)
        torch.cuda.synchronize(dev)
    out = a.output_dir
    with tempfile.TemporaryDirectory() as tmp:
        prof.export_memory_timeline(os.path.join(tmp, "timeline.html"), device=str(dev))
        fig = plt.gcf()
        prof.mem_tl.export_memory_timeline(os.path.join(tmp, "timeline.json"), str(dev))
        _, sizes = json.load(open(os.path.join(tmp, "timeline.json")))
    tl_peak = max(map(sum, sizes))
    assert tl_peak <= YLIM_MIB * MiB, "timeline exceeds plot Y range"
    ax = fig.axes[0]
    ax.set_ylim(0, YLIM_MIB / 1024)  # plot is in GiB; ticks shown in MiB
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 1024:.0f}"))
    ax.set_ylabel("PyTorch tensor memory (MiB)")
    ax.set_title("")
    ax.set_title(f"SimCLR ResNet18 | {MODE_EN[a.mode]} | 1 training step (batch {BATCH}x2 views)\n"
                 f"peak {tl_peak / MiB:.1f} MiB (without cuDNN workspace {(before + pk_model) / MiB:.1f} MiB)",
                 loc="left")
    fig.set_size_inches(14, 4)
    fig.savefig(os.path.join(out, f"{a.mode}_memory.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    r = {"mode": a.mode, "label": MODE_EN[a.mode], "batch": f"{BATCH}x2 views x3x32x32",
         "weights_bytes": n_par * 4, "allocated_before_step_bytes": before,
         "peak_allocated_bytes": peak, "peak_training_tensors_bytes": before + pk_model,
         "cudnn_workspace_peak_bytes": pk_ws, "timeline_peak_bytes": tl_peak,
         "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
         "gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__}
    json.dump(r, open(os.path.join(out, f"{a.mode}_memory.json"), "w"), indent=1, ensure_ascii=False)
    print(f"[SimCLR ResNet18] {MODE_EN[a.mode]}\n"
          f"  resident before step {before / MiB:.1f} MiB | 1-step training peak {(before + pk_model) / MiB:.1f} MiB "
          f"(excl. cuDNN workspace) | max_memory_allocated {peak / MiB:.1f} MiB", flush=True)


def combine(a):
    meta = [json.load(open(os.path.join(a.output_dir, f"{m}_memory.json"))) for m in CALC_MODES]
    assert all(m["gpu"] == meta[0]["gpu"] and m["batch"] == meta[0]["batch"] for m in meta), "measurement settings differ"
    fig, axes = plt.subplots(len(CALC_MODES), 1, figsize=(14, 4 * len(CALC_MODES)), layout="constrained")
    for ax, m in zip(axes, CALC_MODES):
        ax.imshow(plt.imread(os.path.join(a.output_dir, f"{m}_memory.png")))
        ax.set_axis_off()
    png = os.path.join(a.output_dir, f"{NAME}_train_memory_comparison.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"figure, top to bottom: {', '.join(MODE_EN[m] for m in CALC_MODES)} (shared Y axis) -> {png}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mode", choices=CALC_MODES)
    g.add_argument("--combine", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data", default=None)
    p.add_argument("--output-dir", default=RESULTS)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    combine(a) if a.combine else capture(a)


if __name__ == "__main__":
    main()
