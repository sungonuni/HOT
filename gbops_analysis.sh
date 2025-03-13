for transform_scheme in FP HLQ
do
    python gbops_calculation.py \
            --GPU_USE 0 \
            --RUN_NAME vit_b_${transform_scheme} \
            --MODEL vit_b \
            --BATCH_SIZE 1 \
            --PRETRAINED false \
            --DATASET ImageNet100 \
            --transform_scheme $transform_scheme
done