"""CIFAR-10 fine-tuning of the SimCLR backbone with 10% labels.

nipq_hot: the last --ste_epochs use STE rounding; BN is then folded and the model exported as a W4A4 .bin.

    python finetune.py --init fp32|nipq_hot
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from core import qnet  # noqa: E402
from core.common import (CKPT, FP32_MODEL, LABEL_FRAC, LOGS, MODEL_DIR, NAME, W4A4_MODEL,  # noqa: E402
                         finetune_ckpt, pretrain_ckpt)
from core.data import accuracy, augment, cosine_lr, label_subset, load_cifar10  # noqa: E402
from core.model import build_classifier, deploy_w4a4, make_optimizer, simclr_to_classifier  # noqa: E402
from core.pack import export_packed, import_packed  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--init", required=True, choices=["fp32", "nipq_hot"])
p.add_argument("--frac", type=float, default=LABEL_FRAC, help="label fraction")
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--ste_epochs", type=int, default=10, help="nipq_hot: final STE epochs (included in --epochs)")
p.add_argument("--bs", type=int, default=256)
p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--wd", type=float, default=5e-4)
p.add_argument("--eval_every", type=int, default=10)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="cuda:0")
p.add_argument("--data", default=None, help="CIFAR-10 dir (downloaded to IITP_valid/data if omitted)")
args = p.parse_args()

torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device(args.device)
for d in (CKPT, LOGS, MODEL_DIR):
    os.makedirs(d, exist_ok=True)
xtr, ytr, xte, yte = load_cifar10(dev, args.data)
lab = label_subset(ytr, args.frac, args.seed)
quant = args.init == "nipq_hot"

model = build_classifier("nipq_hot" if quant else "fp32").to(dev)
ck = torch.load(pretrain_ckpt(args.init), map_location="cpu")
assert ck["epoch"] == ck["args"]["epochs"], f"pretraining not finished ({ck['epoch']} epochs)"
missing, unexpected = model.load_state_dict(simclr_to_classifier(ck["state_dict"]), strict=False)
assert not unexpected and all(k.startswith("head.fc.") for k in missing), (missing, unexpected)
if quant:
    model.head.fc.wq.init_mse(model.head.fc.fc.weight)  # new classifier only
opt = make_optimizer(model, args.lr, args.wd, nesterov=True)
n = lab.numel()
spe = n // args.bs
total, step, t0 = spe * args.epochs, 0, time.time()
log = {"args": vars(args), "n_labeled": n, "curve": []}
print(f"[{args.init}] {n:,} labels ({args.frac:.0%}), {args.epochs} epochs x {spe} steps", flush=True)
for ep in range(1, args.epochs + 1):
    if quant:
        qnet.LSQ.noise = ep <= args.epochs - args.ste_epochs
    model.train()
    perm = lab[torch.randperm(n, device=dev)]
    loss_sum = torch.zeros((), device=dev)
    for k in range(spe):
        idx = perm[k * args.bs:(k + 1) * args.bs]
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, total, args.lr)
        loss = F.cross_entropy(model(augment(xtr[idx])), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_sum += loss.detach()
        step += 1
    rec = {"epoch": ep, "loss": loss_sum.item() / spe, "time_s": round(time.time() - t0, 1)}
    if quant:
        rec["phase"] = "nipq" if qnet.LSQ.noise else "ste"
    if ep % args.eval_every == 0 or ep == args.epochs:
        rec["test_acc"] = accuracy(model, xte, yte)
    log["curve"].append(rec)
    print(json.dumps(rec), flush=True)
qnet.LSQ.noise = False

model.eval()
log["acc"] = accuracy(model, xte, yte)
meta = {"init": args.init, "args": vars(args), "n_labeled": n}
if args.init == "fp32":
    torch.save({**meta, "state_dict": model.state_dict(), "acc": log["acc"]}, FP32_MODEL)
    print(f"original FP32 accuracy {log['acc']:.2f}% -> {FP32_MODEL}", flush=True)
else:
    qnet.set_hot(model, False)
    qnet.to_deploy(model)
    log["acc"], log["acc_before_bn_fold"] = accuracy(model, xte, yte), log["acc"]
    torch.save({**meta, "state_dict": model.state_dict(), "acc": log["acc"]}, finetune_ckpt("nipq_hot"))
    size = export_packed(model, W4A4_MODEL)
    back = import_packed(deploy_w4a4().to(dev), W4A4_MODEL)
    diff = max((a - b).abs().max().item() for a, b in zip(model.state_dict().values(), back.state_dict().values()))
    assert diff == 0, f"reloaded .bin differs from trained model ({diff})"
    print(f"W4A4 accuracy {log['acc']:.2f}% (before BN fold {log['acc_before_bn_fold']:.2f}%) -> "
          f"{finetune_ckpt('nipq_hot')}, .bin {size:,} bytes -> {W4A4_MODEL}", flush=True)
log["time_s"] = round(time.time() - t0, 1)
json.dump(log, open(os.path.join(LOGS, f"{NAME}_finetune_{args.init}.json"), "w"), indent=1)
