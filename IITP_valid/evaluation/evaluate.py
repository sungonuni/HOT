"""Step 2 정확도: CIFAR-10 test 10,000장 top-1.

  --mode fp32 : 원본 FP32 모델 (models/simclr_resnet18_fp32.pt)
  --mode w4a4 : 배포 파일 models/simclr_resnet18_w4a4.bin 만으로 실행
                (runtime_packed.py: 가중치·activation 을 4bit 정수 비트열로 두고 실행)

    python evaluate.py --mode fp32 --data <CIFAR-10 폴더>
    python evaluate.py --mode w4a4 --data <CIFAR-10 폴더>
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
    p.add_argument("--data", default=None, help="CIFAR-10 폴더 (없으면 IITP_valid/data 에 자동 다운로드)")
    p.add_argument("--chunk", type=int, default=16, help="경량 모델 런타임: 채널 몇 개씩 복원 (0 = 층 전체)")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"], help="경량 모델 런타임: 복원 정밀도")
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
    name = "원본 FP32" if a.mode == "fp32" else QLABEL + " (배포 파일로 실행)"
    print(f"[{TITLE}] {name} 정확도 {r['acc']:.2f}% ({r['n_test']:,}장)", flush=True)
    json.dump(r, open(os.path.join(a.output_dir, f"{a.mode}_acc.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
