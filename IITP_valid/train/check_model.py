"""Step 1 모델 확인 (ckpt/ 의 사전학습 결과 사용).

 1. 구조       : 원본 FP32 와 경량 NIPQ+HOT 이 같은 SimCLR + ResNet18 인지
                 - 백본이 torchvision ResNet18 (CIFAR stem: conv1 3×3/s1, maxpool 제거, 분류기 제거) 과 같은지:
                   파라미터 수·층 모양, 같은 가중치를 넣었을 때 출력이 같은지
                 - 두 모델의 파라미터가 같은지 (경량 모델에는 양자화기 step size 만 더 있음),
                   두 ckpt 가 같은 설정으로 마지막 에폭까지 학습됐는지
 2. 경량 레이어 : 모든 conv·linear 의 가중치와 모든 activation 에 4bit 양자화기가 붙었는지,
                 모든 conv·linear·BN·ReLU 가 HOT backward 를 쓰는지
 3. 저장 텐서   : 학습 forward 에서 backward 용으로 저장한 텐서의 dtype (saved_tensors_hooks).
                 경량 모델은 activation 을 INT8·1bit 로만 저장하고, FP32 로 저장하는 것은 scale·통계(채널·표본당 1개)와
                 양자화된 가중치 사본(가중치 크기)뿐인지
 4. HOT 변환 정확성 : HOT 의 INT8 양자화를 끄고 Hadamard 변환만 하면 기울기가 정확한지. FP64 로 계산한 기울기 기준,
                 HOT(양자화 끔) 의 상대 오차가 FP32 자체의 반올림 오차의 2배 이내인지
 5. NIPQ 구현 정확성 : 메모리를 줄인 NIPQ 구현(마스크 1bit + 노이즈 다시 생성)이 PyTorch autograd 그대로의 NIPQ 와
                 같은 노이즈에서 같은 출력·기울기(양자화기 step size 포함)를 내는지

    python check_model.py --data <CIFAR-10 폴더>
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torchvision  # noqa: E402

from core import hot, nipq, qnet  # noqa: E402
from core.common import A_BITS, BATCH, MODES, W_BITS, pretrain_ckpt  # noqa: E402
from core.data import load_cifar10, simclr_view  # noqa: E402
from core.model import build_simclr, nt_xent  # noqa: E402
from memory_calc import saved_activations  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EXACT_FACTOR = 2.0   # 4: HOT(양자화 끔) 오차 ≤ 2 × FP32 반올림 오차
NIPQ_TOL = 1e-4      # 5: 두 NIPQ 구현의 기울기 최대 상대 오차 (합 순서 차이만 남음)


def torchvision_resnet18():
    m = torchvision.models.resnet18(weights=None)
    m.conv1, m.maxpool, m.fc = nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.Identity(), nn.Identity()
    return m


def load(mode, dev):
    m = build_simclr(mode)
    ck = torch.load(pretrain_ckpt(mode), map_location="cpu")
    m.load_state_dict(ck["state_dict"])
    return m.to(dev), ck


def is_quant_param(name):
    return name.endswith(".s") and (".wq." in name or ".oq." in name)


@torch.no_grad()
def check_structure(models, ckpts, dev):
    f, h = models["fp32"], models["nipq_hot"]
    tv = torchvision_resnet18().to(dev).double().eval()
    a, b = list(f.backbone.state_dict().items()), list(tv.state_dict().items())
    same_shapes = len(a) == len(b) and all(va.shape == vb.shape for (_, va), (_, vb) in zip(a, b))
    tv.load_state_dict({kb: va for (_, va), (kb, _) in zip(a, b)})  # 같은 학습 가중치를 torchvision 모델에 넣음
    x = torch.randn(8, 3, 32, 32, device=dev, dtype=torch.float64)
    bb = copy.deepcopy(f).double().eval()
    o1, o2 = bb.features(x), tv(x)
    pf = {n: p.shape for n, p in f.named_parameters()}
    ph = {n: p.shape for n, p in h.named_parameters() if not is_quant_param(n)}
    r = {"params_fp32": sum(p.numel() for p in f.parameters()),
         "params_nipq_hot": sum(p.numel() for n, p in h.named_parameters() if not is_quant_param(n)),
         "quantizer_step_sizes_nipq_hot": sum(p.numel() for n, p in h.named_parameters() if is_quant_param(n)),
         "params_torchvision_resnet18_backbone": sum(p.numel() for p in tv.parameters()),
         "params_fp32_backbone": sum(p.numel() for p in f.backbone.parameters()),
         "backbone_same_layer_shapes_as_torchvision": same_shapes,
         "max_rel_output_diff_vs_torchvision": ((o1 - o2).abs().max() / o2.abs().max()).item(),
         "fp32_nipq_hot_same_parameters": pf == ph,
         "epochs": {m: ckpts[m]["epoch"] for m in MODES},
         "trained_to_last_epoch": all(ckpts[m]["epoch"] == ckpts[m]["args"]["epochs"] for m in MODES),
         "same_train_settings": all(ckpts["fp32"]["args"][k] == ckpts["nipq_hot"]["args"][k]
                                    for k in ["epochs", "bs", "lr", "wd", "warmup", "seed"])}
    r["ok"] = (same_shapes and r["max_rel_output_diff_vs_torchvision"] < 1e-9 and r["fp32_nipq_hot_same_parameters"]
               and r["params_fp32_backbone"] == r["params_torchvision_resnet18_backbone"]
               and r["trained_to_last_epoch"] and r["same_train_settings"] and ckpts["nipq_hot"]["quantized"])
    return r


def check_layers(h):
    convs = [m for m in h.modules() if isinstance(m, qnet.FusedConv)]
    lins = [m for m in h.modules() if isinstance(m, qnet.QLinear)]
    hot_mods = [m for m in h.modules() if hasattr(m, "hot")]
    qs = qnet.quantizers(h)
    wq = [q for q in qs if q.kind == "w"]
    aq = [q for q in qs if q.kind == "a"]
    r = {"n_conv": len(convs), "n_linear": len(lins), "n_bn": sum(m.bn is not None for m in convs),
         "n_weight_quantizers": len(wq), "weight_bits": sorted({q.bits for q in wq}),
         "n_act_quantizers": len(aq), "act_bits": sorted({q.bits for q in aq}),
         "n_hot_layers": len(hot_mods), "all_hot": all(m.hot for m in hot_mods),
         "all_quantizers_enabled": all(q.enabled for q in qs),
         "convs_without_weight_quant": sum(m.wq is None for m in convs) + sum(m.wq is None for m in lins),
         "convs_without_output_quant": sum(m.oq is None for m in convs)}
    r["ok"] = (r["weight_bits"] == [W_BITS] and r["act_bits"] == [A_BITS] and r["all_hot"]
               and r["all_quantizers_enabled"] and r["convs_without_weight_quant"] == 0
               and r["convs_without_output_quant"] == 0 and len(wq) == len(convs) + len(lins))
    return r


def check_saved(models, x):
    """FP32 로 저장한 텐서 중 원소 수가 배치(512)보다 많고 모양이 가중치와 다른 것 = FP32 activation.
    scale·BN 통계는 표본·채널당 1개라 해당 없음. 모델 밖 손실(NT-Xent)이 저장하는 텐서는 층 종류 "Loss" 로 따로 둔다."""
    r = {}
    for mode, m in models.items():
        wshapes = {tuple(p.shape) for p in m.parameters()}
        recs = saved_activations(m.train(), x)
        by = {}
        for _, dt, _, nb in recs:
            by[dt] = by.get(dt, 0) + nb
        float_act = [(typ, shape) for typ, dt, shape, _ in recs
                     if dt.startswith("float") and typ != "Loss" and torch.Size(shape).numel() > x.size(0)
                     and tuple(shape) not in wshapes]
        r[mode] = {"bytes_by_dtype": by, "n_saved": len(recs), "layers_saving_float_activation": float_act}
    r["ok"] = not r["nipq_hot"]["layers_saving_float_activation"]
    return r


def grads(m, x):
    m.zero_grad(set_to_none=True)
    out = m(x)
    nt_xent(out).backward()
    return out.detach(), {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}


def max_rel(g, ref):
    return max(((g[n].to(ref[n].dtype) - ref[n]).norm() / ref[n].norm().clamp_min(1e-30)).item() for n in ref)


def check_hot(f, x):
    """같은 가중치·같은 입력에서 기울기 비교. 기준: FP64 로 계산한 원본 모델 기울기."""
    f = copy.deepcopy(f).train()
    buf = {k: v.clone() for k, v in f.state_dict().items()}
    _, g64 = grads(copy.deepcopy(f).double(), x.double())
    _, g32 = grads(copy.deepcopy(f), x)
    h = qnet.set_hot(copy.deepcopy(f), True)
    h.load_state_dict(buf)
    hot.QUANTIZE = False
    try:
        _, gh = grads(h, x)
    finally:
        hot.QUANTIZE = True
    out = {"reference": "float64 original model", "fp32_max_rel_err": max_rel(g32, g64),
           "hadamard_only_max_rel_err": max_rel(gh, g64)}
    out["ok"] = out["hadamard_only_max_rel_err"] <= EXACT_FACTOR * out["fp32_max_rel_err"]
    return out


def check_nipq(h, x):
    """같은 가중치·같은 노이즈(같은 시드)에서 메모리 절감 NIPQ 와 autograd 그대로의 NIPQ 비교 (HOT 은 둘 다 켬)."""
    buf = {k: v.clone() for k, v in h.state_dict().items()}
    res = {}
    qnet.LSQ.noise = True
    try:
        for eff in (True, False):
            m = copy.deepcopy(h).train()
            m.load_state_dict(buf)
            nipq.EFFICIENT = eff
            torch.manual_seed(1)
            res[eff] = grads(m, x)
    finally:
        nipq.EFFICIENT, qnet.LSQ.noise = True, False
    (o_e, g_e), (o_n, g_n) = res[True], res[False]
    qn = [n for n in g_n if is_quant_param(n)]
    out = {"same_forward_output": bool(torch.equal(o_e, o_n)),
           "max_rel_err_weights": max_rel({n: g_e[n] for n in g_n if n not in qn}, {n: g_n[n] for n in g_n if n not in qn}),
           "max_rel_err_step_sizes": max_rel({n: g_e[n] for n in qn}, {n: g_n[n] for n in qn}),
           "same_params_with_grad": sorted(g_e) == sorted(g_n), "n_step_size_grads": len(qn)}
    out["ok"] = (out["same_forward_output"] and out["same_params_with_grad"]
                 and max(out["max_rel_err_weights"], out["max_rel_err_step_sizes"]) <= NIPQ_TOL)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data", default=None)
    p.add_argument("--output-dir", default=RESULTS)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    dev = torch.device(a.device)
    loaded = {m: load(m, dev) for m in MODES}
    models, ckpts = {m: v[0] for m, v in loaded.items()}, {m: v[1] for m, v in loaded.items()}
    torch.manual_seed(0)
    xtr, *_ = load_cifar10("cpu", a.data)
    xb = xtr[:BATCH].to(dev)
    with torch.no_grad():
        x = torch.cat([simclr_view(xb), simclr_view(xb)])
    qnet.LSQ.noise = True
    saved = check_saved({m: copy.deepcopy(v) for m, v in models.items()}, x)
    qnet.LSQ.noise = False
    r = {"structure": check_structure(models, ckpts, dev), "layers": check_layers(models["nipq_hot"]),
         "saved_tensors": saved, "hot_exactness": check_hot(models["fp32"], x),
         "nipq_exactness": check_nipq(models["nipq_hot"], x)}
    r["ok"] = all(v["ok"] for v in r.values())
    json.dump(r, open(os.path.join(a.output_dir, "check.json"), "w"), indent=1, ensure_ascii=False)
    s, l, sv, g, q = r["structure"], r["layers"], r["saved_tensors"], r["hot_exactness"], r["nipq_exactness"]
    ok = lambda v: "OK" if v else "FAIL"  # noqa: E731
    mb = lambda d: {k: round(v / 2 ** 20, 1) for k, v in d.items()}  # noqa: E731
    print(f"1. 구조            {ok(s['ok'])}: 파라미터 {s['params_fp32']:,} (경량 {s['params_nipq_hot']:,} + 양자화기 step size "
          f"{s['quantizer_step_sizes_nipq_hot']:,}), 백본 torchvision ResNet18 과 같음 "
          f"{s['backbone_same_layer_shapes_as_torchvision']} (출력 차 {s['max_rel_output_diff_vs_torchvision']:.1e}), "
          f"학습 에폭 {s['epochs']}\n"
          f"2. 경량 레이어     {ok(l['ok'])}: conv {l['n_conv']}·linear {l['n_linear']}·BN {l['n_bn']}, 가중치 양자화기 "
          f"{l['n_weight_quantizers']}개 {l['weight_bits']}bit, activation 양자화기 {l['n_act_quantizers']}개 "
          f"{l['act_bits']}bit, HOT backward 층 {l['n_hot_layers']}개 모두 켜짐 {l['all_hot']}\n"
          f"3. 저장 텐서       {ok(sv['ok'])}: 원본 {mb(sv['fp32']['bytes_by_dtype'])} / 경량 "
          f"{mb(sv['nipq_hot']['bytes_by_dtype'])} (MiB), 경량에서 FP32 activation 을 저장한 층 "
          f"{len(sv['nipq_hot']['layers_saving_float_activation'])}개\n"
          f"4. HOT 변환 정확성 {ok(g['ok'])}: FP64 기준 최대 상대 오차 — HOT(양자화 끔) "
          f"{g['hadamard_only_max_rel_err']:.1e}, 원본 FP32 {g['fp32_max_rel_err']:.1e}\n"
          f"5. NIPQ 구현 정확성 {ok(q['ok'])}: autograd 그대로의 NIPQ 와 forward 출력 같음 {q['same_forward_output']}, "
          f"기울기 최대 상대 오차 가중치 {q['max_rel_err_weights']:.1e} / step size {q['max_rel_err_step_sizes']:.1e} "
          f"({q['n_step_size_grads']}개)\n"
          f"모델 확인: {ok(r['ok'])}", flush=True)


if __name__ == "__main__":
    main()
