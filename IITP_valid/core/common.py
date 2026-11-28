"""Shared paths and settings."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
CKPT = os.path.join(ROOT, "train", "ckpt")
LOGS = os.path.join(ROOT, "train", "logs")

NAME = "simclr_resnet18"
MODES = {"fp32": "original FP32", "nipq_hot": "NIPQ+HOT"}
MODE_EN = {"fp32": "original FP32", "nipq_hot": "NIPQ forward + HOT backward",
           "nipq_hot_naive": "NIPQ (plain autograd) + HOT"}
W_BITS, A_BITS = 4, 4
BATCH = 256  # images per SimCLR step (two views each)
LABEL_FRAC = 0.1
TRAIN_TARGET = 30.0  # required train memory reduction (%)
INFER_TARGET = 85.0  # required inference memory reduction (%)
MiB = 2 ** 20


def pretrain_ckpt(mode):
    return os.path.join(CKPT, f"{NAME}_pretrain_{mode}.pt")


def finetune_ckpt(init):
    return os.path.join(CKPT, f"{NAME}_finetune_{init}.pt")


FP32_MODEL = os.path.join(MODEL_DIR, f"{NAME}_fp32.pt")
W4A4_MODEL = os.path.join(MODEL_DIR, f"{NAME}_w4a4.bin")
