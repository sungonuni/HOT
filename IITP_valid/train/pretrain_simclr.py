"""SimCLR + ResNet18 자기지도 사전학습 (CIFAR-10 train 50,000장, 라벨 사용 안 함, 처음부터 학습).
원본 FP32 와 경량 NIPQ+HOT 은 이 파일 하나로, --mode 만 다르고 나머지 설정(증강·배치·옵티마이저·에폭·시드)은 같다.

  --mode fp32     : FP32 forward + PyTorch 기본 backward
  --mode nipq_hot : 처음 --quant_start 에폭은 양자화기를 끄고 HOT backward 만 쓴다 (무작위 초기화 모델은 step size 를
                    보정할 activation 분포가 없음). 그 뒤 보정 이미지 512장으로 step size 를 초기화하고,
                    남은 에폭은 NIPQ forward (W4A4 의사양자화 노이즈) + HOT INT8 backward 로 학습한다.

    python pretrain_simclr.py --mode fp32     --data <CIFAR-10 폴더>  → ckpt/simclr_resnet18_pretrain_fp32.pt
    python pretrain_simclr.py --mode nipq_hot --data <CIFAR-10 폴더>  → ckpt/simclr_resnet18_pretrain_nipq_hot.pt
    (로그: logs/simclr_resnet18_pretrain_{mode}.json)
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
p.add_argument("--warmup", type=int, default=10, help="learning rate warmup 에폭")
p.add_argument("--quant_start", type=int, default=10, help="nipq_hot: 몇 에폭 뒤에 NIPQ 양자화를 켤지")
p.add_argument("--knn_every", type=int, default=10, help="몇 에폭마다 kNN 정확도를 기록할지 (test set, 모니터링용)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="cuda:0")
p.add_argument("--data", default=None, help="CIFAR-10 폴더 (없으면 IITP_valid/data 에 자동 다운로드)")
args = p.parse_args()

torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False  # 두 모드 모두 FP32 연산 (TF32 끔)
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
        print(f"epoch {ep}: NIPQ W4A4 양자화 시작 (step size 보정 완료)", flush=True)
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
