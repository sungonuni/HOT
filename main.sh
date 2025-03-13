python ViT_main.py \
    --GPU_USE 0 \
    --RUN_NAME vit_b_cifar100_HLQ \
    --DEBUG_MODE false \
    --MODEL Q_vit_b \
    --PRETRAINED true \
    --DATASET cifar100 \
    --AMP false \
    --EPOCHS 50 \
    --BATCH_SIZE 128 \
    --LR 2.5e-5 \
    --SEED 2024 \
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
