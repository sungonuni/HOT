"""Step 2 shared paths, settings and model loading."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from core.common import A_BITS, FP32_MODEL, INFER_TARGET as TARGET, MiB, NAME, W_BITS, W4A4_MODEL  # noqa: E402,F401
from core.model import deploy_w4a4  # noqa: E402,F401
from core.qnet import build_model  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TITLE = "SimCLR ResNet18"
QLABEL = f"W{W_BITS}A{A_BITS}"


def load_fp32(dev="cpu", fold=False):
    """fold=True: BN folded into conv."""
    m = build_model()
    m.load_state_dict(torch.load(FP32_MODEL, map_location="cpu")["state_dict"])
    if fold:
        m.fold_bn()
    return m.to(dev).eval()
