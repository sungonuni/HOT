"""Step 2 accuracy: CIFAR-10 test top-1 (fp32 checkpoint or packed W4A4 .bin).

    python evaluate.py --mode fp32
    python evaluate.py --mode w4a4
"""
import argparse
import json
import os
import time

import torch

from common import QLABEL, RESULTS, TITLE, W4A4_MODEL, deploy_w4a4, load_fp32
from core.data import load_cifar10, to_input


@torch.inference_mode()
def predict(fn, x_u8, dev, bs=1000):
    return torch.cat([fn(to_input(x_u8[i:i + bs]).to(dev)).argmax(1).cpu() for i in range(0, x_u8.size(0), bs)])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["fp32", "w4a4"])
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data", default=None, help="CIFAR-10 dir (downloaded to IITP_valid/data if missing)")
    p.add_argument("--chunk", type=int, default=16, help="W4A4 runtime: channels dequantized per chunk (0 = whole layer)")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"], help="W4A4 runtime: dequantization dtype")
    p.add_argument("--output-dir", default=RESULTS)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    _, _, xte, yte = load_cifar10("cpu", a.data)
    t0 = time.time()
    r = {"model": TITLE, "mode": a.mode, "device": a.device, "n_test": int(yte.numel())}
    if a.mode == "fp32":
        pred = predict(load_fp32(a.device), xte, a.device)
    else:
        from runtime_packed import PackedRuntime
        rt = PackedRuntime(deploy_w4a4(), W4A4_MODEL, a.device, chunk=a.chunk or None,
                           dtype=torch.float32 if a.dtype == "fp32" else torch.bfloat16)
        pred = predict(rt, xte, a.device)
        r["runtime"] = {"chunk": a.chunk or "layer", "dtype": a.dtype}
    r["acc"] = (pred == yte).float().mean().item() * 100
    r["seconds"] = time.time() - t0
    name = "original FP32" if a.mode == "fp32" else QLABEL + " (run from .bin)"
    print(f"[{TITLE}] {name} accuracy {r['acc']:.2f}% ({r['n_test']:,} images)", flush=True)
    json.dump(r, open(os.path.join(a.output_dir, f"{a.mode}_acc.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
