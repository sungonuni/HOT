"""SimCLR + ResNet18 self-supervised pretraining on CIFAR-10 (no labels).

nipq_hot: quantizers are off for the first --quant_start epochs (HOT backward only), then step sizes
are calibrated and training continues with NIPQ W4A4 forward + HOT INT8 backward.

    python pretrain_simclr.py --mode fp32|nipq_hot
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from core import qnet  # noqa: E402
from core.common import BATCH, CKPT, LOGS, MODES, NAME, pretrain_ckpt  # noqa: E402
from core.data import cosine_lr, load_cifar10, to_input  # noqa: E402
from core.model import build_simclr, features, knn_acc, make_optimizer, train_step  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--mode", required=True, choices=list(MODES))
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--bs", type=int, default=BATCH)
p.add_argument("--lr", type=float, default=0.12)
p.add_argument("--wd", type=float, default=5e-4)
p.add_argument("--warmup", type=int, default=10, help="LR warmup epochs")
p.add_argument("--quant_start", type=int, default=10, help="nipq_hot: epoch at which NIPQ quantization starts")
p.add_argument("--knn_every", type=int, default=10, help="kNN accuracy logging interval in epochs (test set, monitoring only)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="cuda:0")
p.add_argument("--data", default=None, help="CIFAR-10 dir (downloaded to IITP_valid/data if omitted)")
args = p.parse_args()

torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device(args.device)
os.makedirs(CKPT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)
xtr, ytr, xte, yte = load_cifar10(dev, args.data)
N = xtr.size(0)

model = build_simclr(args.mode).to(dev)
quant = args.mode == "nipq_hot"
if quant:
    qnet.set_quant(model, False)
opt = make_optimizer(model, args.lr, args.wd)
spe = N // args.bs
total, step, t0 = spe * args.epochs, 0, time.time()
log = {"args": vars(args), "curve": []}
for ep in range(1, args.epochs + 1):
    if quant and ep == args.quant_start + 1:
        qnet.calibrate(model, to_input(xtr[torch.randperm(N, device=dev)[:512]]))
        qnet.LSQ.noise = True
        print(f"epoch {ep}: NIPQ W4A4 quantization on (step sizes calibrated)", flush=True)
    model.train()
    perm = torch.randperm(N, device=dev)
    loss_sum = torch.zeros((), device=dev)
    for k in range(spe):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, total, args.lr, args.warmup * spe)
        loss_sum += train_step(model, opt, xtr[perm[k * args.bs:(k + 1) * args.bs]])
        step += 1
    rec = {"epoch": ep, "loss": loss_sum.item() / spe, "time_s": round(time.time() - t0, 1),
           "quantized": quant and ep > args.quant_start}
    if ep % args.knn_every == 0 or ep == args.epochs:
        rec["knn_acc"] = knn_acc(features(model, xtr), ytr, features(model, xte), yte)
    log["curve"].append(rec)
    print(json.dumps(rec), flush=True)
    torch.save({"state_dict": model.state_dict(), "mode": args.mode, "epoch": ep, "args": vars(args),
                "quantized": rec["quantized"]}, pretrain_ckpt(args.mode))
    json.dump(log, open(os.path.join(LOGS, f"{NAME}_pretrain_{args.mode}.json"), "w"), indent=1)
print(f"done: {pretrain_ckpt(args.mode)}")
