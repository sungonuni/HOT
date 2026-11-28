"""SimCLR (Chen et al., ICML 2020) + ResNet18 백본, CIFAR-10 분류 모델, 학습 모드별 모델 만들기.

학습 모드
  fp32     : 원본. FP32 forward, PyTorch 기본 backward
  nipq_hot : 경량. forward 는 NIPQ 의사양자화 (가중치·activation 4bit, W4A4), backward 는 HOT INT8
             (+ NIPQ 양자화기의 backward 는 마스크 1bit·노이즈 시드만 저장, nipq.py)
두 모드는 같은 구조·같은 파라미터를 쓴다 (경량 모드에는 양자화기 step size 만 더 있음).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import qnet
from .common import A_BITS, W_BITS
from .data import simclr_view, to_input

FEAT_DIM, PROJ_DIM, TEMPERATURE = 512, 128, 0.5


class SimCLR(nn.Module):
    """백본 ResNet18 (분류기 없음) → global average pool (512) → projection head Linear(512,512)-ReLU-Linear(512,128)."""

    def __init__(self):
        super().__init__()
        self.backbone = qnet.ResNet18(num_classes=None)
        self.gap = qnet.QGap()
        self.proj = nn.Sequential(qnet.QLinear(FEAT_DIM, FEAT_DIM), qnet.QReLU(), qnet.QLinear(FEAT_DIM, PROJ_DIM))

    def features(self, x):
        return self.gap(self.backbone(x))

    def forward(self, x):
        return self.proj(self.features(x))


def lightweight(model):
    """경량 모드: 모든 층에 W4A4 양자화기 부착 + HOT backward."""
    return qnet.set_hot(qnet.attach_quantizers_uniform(model, A_BITS, W_BITS), True)


def build_simclr(mode):
    m = SimCLR()
    return lightweight(m) if mode.startswith("nipq_hot") else m


def build_classifier(mode, num_classes=10):
    m = qnet.build_model(num_classes)
    return lightweight(m) if mode.startswith("nipq_hot") else m


def simclr_to_classifier(sd):
    """SimCLR state_dict → 분류 모델 state_dict 일부 (백본 + GAP 양자화기). 분류기 fc 는 새로 초기화."""
    out = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    out.update({"head." + k: v for k, v in sd.items() if k.startswith("gap.")})
    return out


def nt_xent(z, temperature=TEMPERATURE):
    """NT-Xent: z 는 [view1; view2] (2N × d). 같은 이미지의 다른 view 가 양성, 나머지 2N−2 개가 음성."""
    qnet.TAG[0] = "Loss"
    z = F.normalize(z, dim=1)
    n = z.size(0) // 2
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def make_optimizer(model, lr, wd=5e-4, nesterov=False):
    """SGD momentum 0.9. weight decay 는 conv·linear 가중치에만 (BN·bias·양자화기 step size 는 제외)."""
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    return torch.optim.SGD([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0}],
                           lr=lr, momentum=0.9, nesterov=nesterov)


def train_step(model, opt, x_uint8):
    """SimCLR 학습 1 step (학습·메모리 측정이 같은 함수를 씀): 두 view 증강 → forward → NT-Xent → backward → SGD."""
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        x = torch.cat([simclr_view(x_uint8), simclr_view(x_uint8)])
    loss = nt_xent(model(x))
    loss.backward()
    opt.step()
    return loss.detach()


@torch.no_grad()
def features(model, x_uint8, bs=1000):
    model.eval()
    return torch.cat([model.features(to_input(x_uint8[k:k + bs])) for k in range(0, x_uint8.size(0), bs)])


@torch.no_grad()
def knn_acc(ftr, ytr, fte, yte, k=200, t=0.1, classes=10):
    """가중 kNN (Wu et al., 2018): 코사인 유사도 상위 k 개 train 특징의 exp(sim/t) 가중 투표 (사전학습 모니터링용)."""
    ftr, fte = F.normalize(ftr, dim=1), F.normalize(fte, dim=1)
    correct = 0
    for i in range(0, fte.size(0), 1000):
        sim, idx = (fte[i:i + 1000] @ ftr.t()).topk(k, dim=1)
        votes = torch.zeros(sim.size(0), classes, device=fte.device)
        votes.scatter_add_(1, ytr[idx], (sim / t).exp())
        correct += (votes.argmax(1) == yte[i:i + 1000]).sum().item()
    return 100.0 * correct / fte.size(0)


def deploy_w4a4():
    """배포 형태 경량 분류 모델 구조 (BN 을 conv 에 합친 구조 + W4A4 양자화기). 값은 비어 있고 배포 파일로 채운다."""
    return qnet.attach_quantizers_uniform(qnet.build_model().fold_bn(), A_BITS, W_BITS).eval()
