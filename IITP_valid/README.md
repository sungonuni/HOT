# 경량 자가지도 학습 모델의 경량 전 모델 대비 학습·추론 메모리 절감 검증

## 개요 및 성능 지표

경량 자기지도 학습 모델(SimCLR + ResNet18)을 경량 전 모델과 비교하여 두 단계로 나누어 평가한다.

- 원본: 표준 32-bit 부동소수점 학습

- 경량: forward (NIPQ 4bit 의사양자화 (W4A4)), backward: (HOT INT8)

- Step 1. 학습 평가: 학습 메모리 절감률 (목표 30% 이상),

- Step 2. 추론 평가: Step 1 에서 만든 W4A4 배포 파일의 추론 메모리 절감률 (목표 85% 이상)

## 실행 가이드

```
# 의존성 패키지 설치
pip install -r requirements.txt
```

### Step 1. 학습 및 학습 평가

#### 1-1. 모델 학습 (`train/`)

```
cd train

### 1. SimCLR 사전학습 (200 epochs, 라벨 미사용)

python pretrain_simclr.py --mode fp32
# 출력: ckpt/simclr_resnet18_pretrain_fp32.pt

python pretrain_simclr.py --mode nipq_hot
# 출력: ckpt/simclr_resnet18_pretrain_nipq_hot.pt

### 2. CIFAR-10 fine-tuning (라벨 1/10, 100 epochs)

python finetune.py --init fp32
# 출력: ../models/simclr_resnet18_fp32.pt

python finetune.py --init nipq_hot
# 출력: ../models/simclr_resnet18_w4a4.bin, ckpt/simclr_resnet18_finetune_nipq_hot.pt
```

#### 1-2. 학습 평가

1-1 의 출력 파일이 모두 있어야 실행된다.

- `train/ckpt/simclr_resnet18_pretrain_fp32.pt`, `train/ckpt/simclr_resnet18_pretrain_nipq_hot.pt`

- `models/simclr_resnet18_fp32.pt`, `train/ckpt/simclr_resnet18_finetune_nipq_hot.pt`

```
python run_all.py --device cuda:0
```

생성 결과물 (`train/results/`)

- `summary.txt`: 정확도, 학습 메모리 계산값·실측값 비교표 및 PASS/FAIL 판정

- `summary.json`: 세부 수치

- `simclr_resnet18_train_memory_comparison.png`: 학습 1 step GPU 메모리 타임라인


### Step 2. 추론 평가

Step 1 이 만든 `models/simclr_resnet18_fp32.pt` 와 `models/simclr_resnet18_w4a4.bin` 이 있어야 실행된다.

```
cd evaluation
python run_all.py --device cuda:0
```

생성 결과물 (`evaluation/results/`)

- `summary.txt`: 정확도, 추론 메모리 계산값·실측값 비교표 및 PASS/FAIL 판정

- `summary.json`: 세부 수치

- `simclr_resnet18_infer_memory_comparison.png`: 추론 GPU 메모리 타임라인


## 검증 결과 요약

- 측정 환경: NVIDIA L40S, PyTorch `2.12.1+cu130`

### 1. Step 1 학습 평가

| 평가 항목 | 원본 (FP32) | 경량 (NIPQ + HOT) | 절감률 | 판정 |
| ----- | ----- | ----- | ----- | ----- |
| Fine-tuning 정확도 (라벨 1/10) | 84.55% | 83.65% | -0.90%p | \- |
| 학습 메모리 계산값 | 2429.2 MiB | 945.4 MiB | 61.1% | PASS |
| 학습 메모리 실측 | 2420.4 MiB | 1350.5 MiB | 44.2% | PASS |

### 2. Step 2 추론 평가

| 평가 항목 | 원본 (FP32) | 경량 (W4A4 배포 파일) | 절감률 | 판정 |
| ----- | ----- | ----- | ----- | ----- |
| 정확도 | 84.55% | 83.64% | -0.91%p | \- |
| 추론 메모리 계산값 | 43.357 MiB | 5.454 MiB | 87.4% | PASS |
| 추론 메모리 실측 | 43.621 MiB | 5.669 MiB | 87.0% | PASS |
