"""Step 2 모델 확인 (Step 1 이 만든 models/ 의 파일만 사용).

 1. 구조   : 원본 models/simclr_resnet18_fp32.pt 가 torchvision ResNet18 (CIFAR stem, 10 클래스) 과 같은지
             (파라미터 수, 층 모양, 같은 가중치에서 출력 일치)
 2. 배포 파일 models/simclr_resnet18_w4a4.bin
    - 파일 크기 = 모델 구조와 비트수로 계산한 가중치 메모리 (가중치 수 × 비트/8 + 채널별 scale·bias + activation scale)
    - 모델 구조 순서대로 읽었을 때 남거나 모자라는 바이트가 없는지
    - 가중치가 모두 4bit 정수 격자 위인지 ([-7, 7]) × 채널별 scale, activation 양자화기가 모두 4bit 인지
    - 읽은 모델을 다시 저장하면 원래 파일과 바이트 단위로 같은지

    python check_model.py
"""
import argparse
import filecmp
import json
import os
import tempfile

import torch
import torch.nn as nn
import torchvision

from common import A_BITS, QLABEL, RESULTS, TITLE, W_BITS, W4A4_MODEL, deploy_w4a4, load_fp32
from core.pack import export_packed, import_packed
from core.qnet import LSQ, FusedConv, QLinear, weight_bytes


def torchvision_model():
    """torchvision ResNet18 + CIFAR(32×32) stem + 10 클래스 분류기."""
    m = torchvision.models.resnet18(weights=None, num_classes=10)
    m.conv1, m.maxpool = nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.Identity()
    return m.eval()


@torch.no_grad()
def check_structure():
    fp32, tv = load_fp32().double(), torchvision_model().double()
    a, b = list(fp32.state_dict().items()), list(tv.state_dict().items())
    same_shapes = len(a) == len(b) and all(va.shape == vb.shape for (_, va), (_, vb) in zip(a, b))
    tv.load_state_dict({kb: va for (_, va), (kb, _) in zip(a, b)})  # 같은 학습 가중치를 torchvision 모델에 넣음
    x = torch.randn(4, 3, 32, 32, dtype=torch.float64)
    o1, o2 = fp32(x), tv(x)
    return {"params_ours": sum(p.numel() for p in fp32.parameters()),
            "params_torchvision": sum(p.numel() for p in tv.parameters()),
            "same_layer_shapes": same_shapes,
            "max_rel_output_diff": ((o1 - o2).abs().max() / o2.abs().max()).item()}


@torch.no_grad()
def check_bits(m):
    wq = []
    off, n_layers, n_w, lo, hi = 0, 0, 0, 0, 0
    for mod in m.modules():
        if isinstance(mod, (FusedConv, QLinear)):
            w = mod.conv.weight if isinstance(mod, FusedConv) else mod.fc.weight
            s = mod.wq.s.abs().view((-1,) + (1,) * (w.dim() - 1))
            q = mod.wq.int_values(w)
            off += int(((q * s - w).abs() > 1e-6 * s).sum())
            lo, hi = min(lo, int(q.min())), max(hi, int(q.max()))
            n_layers += 1
            n_w += w.numel()
            wq.append(mod.wq)
    aq = [q for q in m.modules() if isinstance(q, LSQ) and all(q is not x for x in wq)]
    return {"n_weight_quantizers": len(wq), "weight_bits_used": sorted({q.bits for q in wq}),
            "n_act_quantizers": len(aq), "act_bits_used": sorted({q.bits for q in aq}),
            "n_weight_layers": n_layers, "n_weights": n_w, "weights_off_grid": off, "weight_int_range": [lo, hi]}


def check_quant():
    path = W4A4_MODEL
    size = os.path.getsize(path)
    calc = weight_bytes(deploy_w4a4())
    q = import_packed(deploy_w4a4(), path)  # 크기가 안 맞으면 여기서 오류
    ck = check_bits(q)
    with tempfile.TemporaryDirectory() as tmp:
        again = os.path.join(tmp, "again.bin")
        export_packed(q, again)
        same = filecmp.cmp(path, again, shallow=False)
    pf = {"file": os.path.basename(path), "file_bytes": size, "weight_bytes_calc": calc,
          "size_match": size == calc, "resave_identical": same}
    ok = (ck["weight_bits_used"] == [W_BITS] and ck["act_bits_used"] == [A_BITS] and ck["weights_off_grid"] == 0
          and pf["size_match"] and same)
    print(f"  [{QLABEL}] 파일 models/{pf['file']}: {size:,} bytes, 구조·비트수로 계산한 가중치 메모리 {calc:,} bytes "
          f"(일치 {pf['size_match']})\n"
          f"         가중치 양자화기 {ck['n_weight_quantizers']}개 {ck['weight_bits_used']}bit (정수 범위 "
          f"{ck['weight_int_range']}), activation 양자화기 {ck['n_act_quantizers']}개 {ck['act_bits_used']}bit, "
          f"가중치 {ck['n_weights']:,}개 중 정수 격자 밖 {ck['weights_off_grid']}개, 다시 저장한 파일과 같음 {same}"
          f"  → {'OK' if ok else 'FAIL'}", flush=True)
    return {"bits": ck, "packed_file": pf, "ok": ok}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=RESULTS)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    st = check_structure()
    st_ok = (st["params_ours"] == st["params_torchvision"] and st["same_layer_shapes"]
             and st["max_rel_output_diff"] < 1e-9)
    print(f"[{TITLE}]\n"
          f"  구조: torchvision 과 파라미터 {st['params_ours']:,} vs {st['params_torchvision']:,}, 층 모양 일치 "
          f"{st['same_layer_shapes']}, 같은 가중치에서 출력 상대오차 {st['max_rel_output_diff']:.1e}  "
          f"→ {'OK' if st_ok else 'FAIL'}", flush=True)
    r = {"model": TITLE, "structure": st, "structure_ok": st_ok, "quant": check_quant()}
    r["ok"] = st_ok and r["quant"]["ok"]
    json.dump(r, open(os.path.join(a.output_dir, "check.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
