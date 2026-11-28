"""Step 2 analytic inference memory (weights + peak activation, batch 1), FP32 (BN folded) vs W4A4.

    python memory_calc.py
"""
import argparse
import json
import os

import torch

from common import QLABEL, RESULTS, TARGET, TITLE, MiB, deploy_w4a4, load_fp32
from core.qnet import peak_activation_bytes, weight_bytes


def memory(m):
    w = weight_bytes(m)
    a, at = peak_activation_bytes(m.cpu(), torch.randn(1, 3, 32, 32))
    return {"weight_bytes": w, "peak_activation_bytes": a, "total_bytes": w + a, "peak_at": at}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=RESULTS)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    f, q = memory(load_fp32(fold=True)), memory(deploy_w4a4())
    red = 100 * (1 - q["total_bytes"] / f["total_bytes"])
    r = {"model": TITLE, "input": "1x3x32x32", "fp32": f, "w4a4": {**q, "reduction_pct": red, "pass": red >= TARGET}}
    lines = [f"[{TITLE}] calculated (NPU integer ops, 32x32 input, 1 image, MiB)",
             f"  {'':16s}{'FP32':>10s}{QLABEL:>10s}"]
    for label, key in [("weights", "weight_bytes"), ("peak activation", "peak_activation_bytes"), ("total", "total_bytes")]:
        lines.append(f"  {label:16s}{f[key] / MiB:10.3f}{q[key] / MiB:10.3f}")
    lines.append(f"  {QLABEL} reduction {red:.1f}% {'PASS' if red >= TARGET else 'FAIL'}")
    print("\n".join(lines), flush=True)
    json.dump(r, open(os.path.join(a.output_dir, "memory_calc.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
