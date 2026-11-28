"""SimCLR + ResNet18 models. fp32: baseline; nipq_hot: NIPQ W4A4 forward + HOT INT8 backward."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import qnet
from .common import A_BITS, W_BITS
from .data import simclr_view, to_input

FEAT_DIM, PROJ_DIM, TEMPERATURE = 512, 128, 0.5


class SimCLR(nn.Module):
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
    return qnet.set_hot(qnet.attach_quantizers_uniform(model, A_BITS, W_BITS), True)


def build_simclr(mode):
    m = SimCLR()
    return lightweight(m) if mode.startswith("nipq_hot") else m


def build_classifier(mode, num_classes=10):
    m = qnet.build_model(num_classes)
    return lightweight(m) if mode.startswith("nipq_hot") else m


def simclr_to_classifier(sd):
    """Map SimCLR backbone/GAP weights to the classifier; fc stays freshly initialized."""
    out = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    out.update({"head." + k: v for k, v in sd.items() if k.startswith("gap.")})
    return out


def nt_xent(z, temperature=TEMPERATURE):
    """NT-Xent loss; z = [view1; view2]."""
    qnet.TAG[0] = "Loss"
    z = F.normalize(z, dim=1)
    n = z.size(0) // 2
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def make_optimizer(model, lr, wd=5e-4, nesterov=False):
    """SGD; weight decay only on conv/linear weights."""
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    return torch.optim.SGD([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0}],
                           lr=lr, momentum=0.9, nesterov=nesterov)


def train_step(model, opt, x_uint8):
    """One SimCLR step (shared by training and memory profiling)."""
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
    """Weighted kNN accuracy (Wu et al., 2018) for pretraining monitoring."""
    ftr, fte = F.normalize(ftr, dim=1), F.normalize(fte, dim=1)
    correct = 0
    for i in range(0, fte.size(0), 1000):
        sim, idx = (fte[i:i + 1000] @ ftr.t()).topk(k, dim=1)
        votes = torch.zeros(sim.size(0), classes, device=fte.device)
        votes.scatter_add_(1, ytr[idx], (sim / t).exp())
        correct += (votes.argmax(1) == yte[i:i + 1000]).sum().item()
    return 100.0 * correct / fte.size(0)


def deploy_w4a4():
    """Empty BN-folded W4A4 classifier, to be filled from the .bin file."""
    return qnet.attach_quantizers_uniform(qnet.build_model().fold_bn(), A_BITS, W_BITS).eval()
