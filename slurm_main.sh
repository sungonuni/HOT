#!/bin/sh
#SBATCH -J  hot_fp4    # Job Name
#SBATCH -o  tunnel_history/%x.%j.out    # Name of stdout output file (%j expands to %jobId)
#SBATCH -e  tunnel_history/%x.%j.err   # Name of stderr output file
#SBATCH -t  3-00:00:00        # Run time (hh:mm:ss)
# #SBATCH --nodelist=n[51,52,53-59] 여기에 node list 지정. 반드시 아래에서 지정한 partition에 포함되는 node들이어야 함.

#### Select  GPU
#SBATCH -p A100-80GB           # queue  name  or  partiton
#SBATCH -n 1              # number of nodes
#SBATCH -N 1
#SBATCH --cpus-per-task=4 # cpu 수 지정
#SBATCH -q hpgpu

# >>> Number of GPUs <<< #
#SBATCH --gres=gpu:1   # gpus per node

python ViT_main.py \
    --GPU_USE 0 \
    --RUN_NAME vit_b_cifar100_HOT_giMXFP4_gwINT8 \
    --DEBUG_MODE false \
    --WANDB true \
    --WANDB_PROJECT Hadamard_Quant \
    --MODEL Q_vit_b \
    --PRETRAINED true \
    --DATASET cifar100 \
    --DATA_DIR /home3/sungonuni/dataset \
    --AMP false \
    --EPOCHS 50 \
    --BATCH_SIZE 128 \
    --LR 2.5e-5 \
    --SEED 2024 \
    \
    --realGEMM false \
    --fakeGEMM true \
    --quantAuto false \
    --quantBWDGogi mxfp4 \
    --quantBWDWgt mxfp4 \
    --quantBWDGogw int \
    --quantBWDAct int \
    \
    --precisionScheduling false \
    --milestone 50 \
    --GogiQuantBit 4 \
    --weightQuantBit 4 \
    --GogwQuantBit 8 \
    --actQuantBit 8 \
    \
    --transform_scheme gih_gwlr \
    --TransformGogi true \
    --TransformWgt true \
    --TransformGogw true \
    --TransformInput true \
