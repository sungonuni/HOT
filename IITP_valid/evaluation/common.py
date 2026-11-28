"""Step 2 공통: 경로·설정·모델 불러오기 (Step 1 이 만든 models/ 의 원본 FP32 ckpt 와 W4A4 배포 파일만 사용)."""
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
    """원본 FP32 모델 (fold=True 면 BN 을 conv 에 합친 배포 형태)."""
    m = build_model()
    m.load_state_dict(torch.load(FP32_MODEL, map_location="cpu")["state_dict"])
    if fold:
        m.fold_bn()
    return m.to(dev).eval()
