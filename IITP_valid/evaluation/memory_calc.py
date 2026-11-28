"""Step 2 추론 메모리 계산값 (NPU 정수 연산 기준): 가중치 + 피크 activation, 이미지 1장. 원본 FP32 vs W4A4.

  가중치          = Σ 층별 (가중치 수 × 비트/8) + 출력 채널별 scale·bias(각 4B) + activation 양자화기별 8B
  피크 activation = 그래프 실행 순서상 동시에 살아 있는 텐서(원소 수 × 비트/8)의 합의 최댓값 (torch.fx)
  계산 기준: NPU 가 정수 × 정수를 32bit 로 누적한 뒤 바로 4bit activation 으로 저장 (실수 복원 사본 없음)
  원본 FP32 도 배포 형태(BN 을 conv 에 합침)로 계산

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
    lines = [f"[{TITLE}] 계산값 (NPU 정수 연산 기준, 입력 32×32, 이미지 1장, MiB)",
             f"  {'':16s}{'원본 FP32':>10s}{QLABEL:>10s}"]
    for label, key in [("가중치", "weight_bytes"), ("피크 activation", "peak_activation_bytes"), ("합계", "total_bytes")]:
        lines.append(f"  {label:16s}{f[key] / MiB:11.3f}{q[key] / MiB:10.3f}")
    lines.append(f"  {QLABEL} 절감률 {red:.1f}% {'PASS' if red >= TARGET else 'FAIL'}")
    print("\n".join(lines), flush=True)
    json.dump(r, open(os.path.join(a.output_dir, "memory_calc.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
