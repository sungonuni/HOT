"""공통 경로·설정 (Step 1 학습 평가, Step 2 추론 평가 공용)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # IITP_valid/
MODEL_DIR = os.path.join(ROOT, "models")  # Step 1 이 만들고 Step 2 가 읽는 최종 모델
CKPT = os.path.join(ROOT, "train", "ckpt")  # 학습 중간 결과 (사전학습·fine-tuning)
LOGS = os.path.join(ROOT, "train", "logs")

NAME = "simclr_resnet18"
MODES = {"fp32": "원본 FP32", "nipq_hot": "경량 NIPQ+HOT"}
MODE_EN = {"fp32": "original FP32", "nipq_hot": "NIPQ forward + HOT backward",
           "nipq_hot_naive": "NIPQ (plain autograd) + HOT"}
W_BITS, A_BITS = 4, 4  # 경량 모델: 가중치 4bit, activation 4bit (W4A4). HOT backward 는 INT8
BATCH = 256            # SimCLR 배치 (이미지 수). 한 step 에 두 view → 512 장
LABEL_FRAC = 0.1       # fine-tuning 라벨 비율 (CIFAR-10 train 의 1/10 = 5,000장, 클래스별 500장)
TRAIN_TARGET = 30.0    # 원본 대비 학습 메모리 절감률 목표 (%)
INFER_TARGET = 85.0    # 원본 대비 추론 메모리 절감률 목표 (%)
MiB = 2 ** 20


def pretrain_ckpt(mode):
    return os.path.join(CKPT, f"{NAME}_pretrain_{mode}.pt")


def finetune_ckpt(init):
    """init: nipq_hot"""
    return os.path.join(CKPT, f"{NAME}_finetune_{init}.pt")


FP32_MODEL = os.path.join(MODEL_DIR, f"{NAME}_fp32.pt")    # 원본: SimCLR FP32 사전학습 → FP32 fine-tuning
W4A4_MODEL = os.path.join(MODEL_DIR, f"{NAME}_w4a4.bin")   # 경량: NIPQ+HOT 사전학습 → fine-tuning → 4bit 배포 파일
