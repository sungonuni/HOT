"""Step 1 정확도: CIFAR-10 test 10,000장 top-1 (test set 은 학습·모델 선택에 쓰지 않음, 마지막 에폭 모델).

  fp32     : 원본. SimCLR FP32 사전학습 → FP32 fine-tuning (라벨 10%)        models/simclr_resnet18_fp32.pt
  nipq_hot : 경량. SimCLR NIPQ+HOT 사전학습 → NIPQ+HOT fine-tuning (라벨 10%) ckpt/simclr_resnet18_finetune_nipq_hot.pt
             (배포 형태: BN 을 합친 W4A4 모델, 실제 반올림. Step 2 는 같은 모델을 배포 파일 .bin 만으로 다시 잰다)

    python evaluate.py --data <CIFAR-10 폴더>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from core.common import FP32_MODEL, ROOT, finetune_ckpt  # noqa: E402
from core.data import accuracy, load_cifar10  # noqa: E402
from core.model import build_classifier, deploy_w4a4  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SOURCES = {"fp32": FP32_MODEL, "nipq_hot": finetune_ckpt("nipq_hot")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data", default=None)
    p.add_argument("--output-dir", default=RESULTS)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(a.device)
    _, _, xte, yte = load_cifar10(dev, a.data)
    r = {}
    for k, path in SOURCES.items():
        ck = torch.load(path, map_location="cpu")
        m = deploy_w4a4() if k == "nipq_hot" else build_classifier("fp32")
        m.load_state_dict(ck["state_dict"])
        r[k] = {"acc": accuracy(m.to(dev), xte, yte), "n_labeled": ck["n_labeled"], "init": ck["init"],
                "epochs": ck["args"]["epochs"], "label_frac": ck["args"]["frac"], "label_seed": ck["args"]["seed"],
                "file": os.path.relpath(path, ROOT)}
        print(f"[{k}] 라벨 {ck['n_labeled']:,}장 → test 정확도 {r[k]['acc']:.2f}%", flush=True)
    r["same_finetune_labels"] = all(r["fp32"][k] == r["nipq_hot"][k] for k in ["label_frac", "label_seed", "epochs"])
    json.dump(r, open(os.path.join(a.output_dir, "acc.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
