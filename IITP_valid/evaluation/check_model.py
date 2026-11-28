"""Step 2 model check: FP32 model matches torchvision ResNet18; W4A4 .bin size, bit widths and round-trip.

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
    """torchvision ResNet18 with CIFAR stem, 10 classes."""
    m = torchvision.models.resnet18(weights=None, num_classes=10)
    m.conv1, m.maxpool = nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.Identity()
    return m.eval()


@torch.no_grad()
def check_structure():
    fp32, tv = load_fp32().double(), torchvision_model().double()
    a, b = list(fp32.state_dict().items()), list(tv.state_dict().items())
    same_shapes = len(a) == len(b) and all(va.shape == vb.shape for (_, va), (_, vb) in zip(a, b))
    tv.load_state_dict({kb: va for (_, va), (kb, _) in zip(a, b)})
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
    q = import_packed(deploy_w4a4(), path)  # raises on size mismatch
    ck = check_bits(q)
    with tempfile.TemporaryDirectory() as tmp:
        again = os.path.join(tmp, "again.bin")
        export_packed(q, again)
        same = filecmp.cmp(path, again, shallow=False)
    pf = {"file": os.path.basename(path), "file_bytes": size, "weight_bytes_calc": calc,
          "size_match": size == calc, "resave_identical": same}
    ok = (ck["weight_bits_used"] == [W_BITS] and ck["act_bits_used"] == [A_BITS] and ck["weights_off_grid"] == 0
          and pf["size_match"] and same)
    print(f"  [{QLABEL}] file models/{pf['file']}: {size:,} bytes, weight memory from shapes/bits {calc:,} bytes "
          f"(match {pf['size_match']})\n"
          f"         {ck['n_weight_quantizers']} weight quantizers {ck['weight_bits_used']}bit (int range "
          f"{ck['weight_int_range']}), {ck['n_act_quantizers']} activation quantizers {ck['act_bits_used']}bit, "
          f"{ck['weights_off_grid']} of {ck['n_weights']:,} weights off the int grid, re-export identical {same}"
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
          f"  structure: params {st['params_ours']:,} vs torchvision {st['params_torchvision']:,}, same layer shapes "
          f"{st['same_layer_shapes']}, output rel. error with same weights {st['max_rel_output_diff']:.1e}  "
          f"→ {'OK' if st_ok else 'FAIL'}", flush=True)
    r = {"model": TITLE, "structure": st, "structure_ok": st_ok, "quant": check_quant()}
    r["ok"] = st_ok and r["quant"]["ok"]
    json.dump(r, open(os.path.join(a.output_dir, "check.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
