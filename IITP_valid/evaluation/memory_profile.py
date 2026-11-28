"""Step 2 measured GPU inference memory (batch 1) and memory timeline plot.

peak_model_tensors excludes cuDNN workspace; peak_allocated includes it.

    python memory_profile.py --mode fp32
    python memory_profile.py --mode w4a4
    python memory_profile.py --combine
"""
import argparse
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")
import tempfile  # noqa: E402

_ntf = tempfile.NamedTemporaryFile
# Workaround: export_memory_timeline opens its temp PNG with "wb" and then reads it back.
tempfile.NamedTemporaryFile = lambda mode="w+b", *a, **k: _ntf("w+b" if mode == "wb" else mode, *a, **k)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from common import QLABEL, RESULTS, TITLE, W4A4_MODEL, MiB, deploy_w4a4, load_fp32  # noqa: E402
from core.data import load_cifar10, to_input  # noqa: E402

YLIM_MIB = 80
MODES = ["fp32", "w4a4"]


class Registry(nn.Module):
    """Registers weight storage as parameters so the profiler labels it PARAMETER (no copy)."""

    def __init__(self, tensors):
        super().__init__()
        for i, t in enumerate(tensors):
            self.register_parameter(f"w{i}", nn.Parameter(t.detach(), requires_grad=False))

    def forward(self, x):
        return x


def replay_peak(snapshot, device_index):
    """Replay alloc trace -> (peak, peak without cuDNN workspace, peak cuDNN workspace)."""
    ws, cur_all, cur_model, cur_ws, pk_all, pk_model, pk_ws = set(), 0, 0, 0, 0, 0, 0
    for e in snapshot["device_traces"][device_index]:
        if e["action"] == "alloc":
            is_ws = any("run_conv_plan" in f["name"] or "cudnn" in f["name"].lower() for f in e["frames"])
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
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    _, _, xte, _ = load_cifar10("cpu", a.data)
    if a.mode == "fp32":
        fn = load_fp32(dev, fold=True)
        weights = sum(t.numel() * t.element_size() for t in fn.parameters())
        reg, label = Registry(list(fn.parameters())), "original FP32 (BN folded)"
    else:
        from runtime_packed import PackedRuntime
        fn = PackedRuntime(deploy_w4a4(), W4A4_MODEL, dev,
                           chunk=a.chunk or None, dtype=torch.float32 if a.dtype == "fp32" else torch.bfloat16)
        weights = fn.weight_bytes
        reg = Registry([fn.blob])
        label = f"{QLABEL} packed ({'whole layer' if not a.chunk else f'{a.chunk} ch'} {a.dtype} dequant)"
    x = to_input(xte[:1]).to(dev)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    with torch.inference_mode():
        for _ in range(3):  # warmup
            fn(x)
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        before = torch.cuda.memory_allocated(dev)
        torch.cuda.memory._record_memory_history(max_entries=1_000_000, stacks="all")
        fn(x)
        torch.cuda.synchronize(dev)
        snap = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(enabled=None)
        peak = torch.cuda.max_memory_allocated(dev)
    pk_all, pk_model, pk_ws = replay_peak(snap, idx)
    with torch.inference_mode():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True, with_stack=True) as prof:
            reg(x)
            fn(x)
            torch.cuda.synchronize(dev)
    tag = a.mode
    out = a.output_dir
    with tempfile.TemporaryDirectory() as tmp:
        prof.export_memory_timeline(os.path.join(tmp, "timeline.html"), device=str(dev))
        fig = plt.gcf()
        prof.mem_tl.export_memory_timeline(os.path.join(tmp, "timeline.json"), str(dev))
        _, sizes = json.load(open(os.path.join(tmp, "timeline.json")))
    tl_peak = max(map(sum, sizes))
    assert tl_peak <= YLIM_MIB * MiB, "plot exceeds Y-axis limit"
    ax = fig.axes[0]
    ax.set_ylim(0, YLIM_MIB / 1024)  # plot is in GiB; label ticks in MiB
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 1024:.0f}"))
    ax.set_ylabel("PyTorch tensor memory (MiB)")
    ax.set_title(f"{TITLE} | {label} | peak {tl_peak / MiB:.2f} MiB "
                 f"(without cuDNN workspace {(before + pk_model) / MiB:.2f} MiB)")
    fig.set_size_inches(14, 4)
    fig.savefig(os.path.join(out, f"{tag}_memory.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    r = {"model": TITLE, "mode": a.mode, "label": label, "input": "1x3x32x32",
         "runtime": None if a.mode == "fp32" else {"chunk": a.chunk or "layer", "dtype": a.dtype},
         "weights_resident_bytes": weights, "input_bytes": x.numel() * 4, "allocated_before_run_bytes": before,
         "peak_allocated_bytes": peak, "peak_model_tensors_bytes": before + pk_model,
         "cudnn_workspace_peak_bytes": pk_ws, "timeline_peak_bytes": tl_peak,
         "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
         "gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__}
    json.dump(r, open(os.path.join(out, f"{tag}_memory.json"), "w"), indent=1, ensure_ascii=False)
    print(f"[{TITLE}] {label}\n"
          f"  resident weights {weights / MiB:.3f} MiB | inference peak {(before + pk_model) / MiB:.3f} MiB "
          f"(excl. cuDNN workspace)", flush=True)


def combine(a):
    tags = MODES
    meta = [json.load(open(os.path.join(a.output_dir, f"{t}_memory.json"))) for t in tags]
    assert all(m["gpu"] == meta[0]["gpu"] and m["input"] == meta[0]["input"] for m in meta), "measurement conditions differ"
    fig, axes = plt.subplots(len(tags), 1, figsize=(14, 4 * len(tags)), layout="constrained")
    for ax, t in zip(axes, tags):
        ax.imshow(plt.imread(os.path.join(a.output_dir, f"{t}_memory.png")))
        ax.set_axis_off()
    png = os.path.join(a.output_dir, "simclr_resnet18_infer_memory_comparison.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: original FP32 (top), {QLABEL} (bottom), shared Y axis -> {png}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mode", choices=MODES)
    g.add_argument("--combine", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data", default=None)
    p.add_argument("--chunk", type=int, default=16, help="W4A4: channels dequantized per chunk (0 = whole layer)")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"], help="W4A4: dequantization dtype")
    p.add_argument("--output-dir", default=RESULTS)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    combine(a) if a.combine else capture(a)


if __name__ == "__main__":
    main()
