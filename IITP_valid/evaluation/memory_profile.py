"""Step 2 실제 GPU 추론 메모리 측정 (이미지 1장 추론) + PyTorch 메모리 타임라인 그림 (torch.profiler).
원본 FP32 와 W4A4 를 각각 별도 프로세스로 실행하고, 그림은 같은 Y 축으로 위아래 합친다.

측정값
  - peak_model_tensors : 추론 중 GPU 텐서 메모리 피크 (가중치 + activation + 런타임 임시 버퍼).
                         CUDA 메모리 할당 기록을 시간순으로 다시 더해서 구하고, cuDNN 이 conv 알고리즘용으로
                         잠깐 잡는 작업공간은 뺀다.
  - peak_allocated     : 위 값에 cuDNN 작업공간까지 포함한 값 (torch.cuda.max_memory_allocated)
  cuBLAS 작업공간은 CUBLAS_WORKSPACE_CONFIG=:0:0 으로 끈다.

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
# PyTorch export_memory_timeline(html) 버그 우회: 임시 PNG 를 "wb" 로 열고 다시 읽음 → "w+b" 로 열게 함 (그림 내용 무관)
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
    """프로파일러가 가중치 저장공간을 PARAMETER 로 분류하도록 같은 저장공간을 가리키는 별칭만 등록 (복사 없음)."""

    def __init__(self, tensors):
        super().__init__()
        for i, t in enumerate(tensors):
            self.register_parameter(f"w{i}", nn.Parameter(t.detach(), requires_grad=False))

    def forward(self, x):
        return x


def replay_peak(snapshot, device_index):
    """할당 기록을 시간순으로 다시 더함: (전체 피크, cuDNN 작업공간 제외 피크, 동시에 잡힌 cuDNN 작업공간 최대)."""
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
        for _ in range(3):  # 워밍업: cuDNN 핸들 생성·알고리즘 선택
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
    # PyTorch 메모리 타임라인 (그림용, 별도 실행 1회)
    with torch.inference_mode():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True, with_stack=True) as prof:
            reg(x)
            fn(x)
            torch.cuda.synchronize(dev)
    tag = a.mode
    out = a.output_dir
    with tempfile.TemporaryDirectory() as tmp:  # PyTorch 출력(html·원시 json)은 그림에만 쓰고 버림
        prof.export_memory_timeline(os.path.join(tmp, "timeline.html"), device=str(dev))  # 그림을 그림
        fig = plt.gcf()
        prof.mem_tl.export_memory_timeline(os.path.join(tmp, "timeline.json"), str(dev))
        _, sizes = json.load(open(os.path.join(tmp, "timeline.json")))
    tl_peak = max(map(sum, sizes))
    assert tl_peak <= YLIM_MIB * MiB, "그림 Y 축 범위를 넘음"
    ax = fig.axes[0]
    ax.set_ylim(0, YLIM_MIB / 1024)  # PyTorch 그림 단위는 GiB → 눈금만 MiB 로 표시
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
          f"  가중치 상주 {weights / MiB:.3f} MiB | 추론 중 피크 {(before + pk_model) / MiB:.3f} MiB "
          f"(cuDNN 작업공간 제외)", flush=True)


def combine(a):
    tags = MODES
    meta = [json.load(open(os.path.join(a.output_dir, f"{t}_memory.json"))) for t in tags]
    assert all(m["gpu"] == meta[0]["gpu"] and m["input"] == meta[0]["input"] for m in meta), "측정 조건 불일치"
    fig, axes = plt.subplots(len(tags), 1, figsize=(14, 4 * len(tags)), layout="constrained")
    for ax, t in zip(axes, tags):
        ax.imshow(plt.imread(os.path.join(a.output_dir, f"{t}_memory.png")))
        ax.set_axis_off()
    png = os.path.join(a.output_dir, "simclr_resnet18_infer_memory_comparison.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"그림: 위에서부터 원본 FP32, {QLABEL} (같은 Y 축) → {png}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mode", choices=MODES)
    g.add_argument("--combine", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data", default=None)
    p.add_argument("--chunk", type=int, default=16, help="경량 모델: 채널 몇 개씩 복원 (0 = 층 전체)")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"], help="경량 모델: 복원 정밀도")
    p.add_argument("--output-dir", default=RESULTS)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    combine(a) if a.combine else capture(a)


if __name__ == "__main__":
    main()
