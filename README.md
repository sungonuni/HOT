# \[CVPR 2025\] HOT: Hadamard-based Optimized Training
This repositorty is official code of HOT: Hadamard-based Optimized Training.


### Overview
It has become increasingly important to optimize backpropagation to reduce memory usage and computational overhead. Achieving this goal is highly challenging, as multiple objectives must be considered jointly while maintaining training quality. In this paper, we focus on matrix multiplication, which accounts for the largest portion of training costs, and analyze its backpropagation in detail to identify lightweight techniques that offer the best benefits. Based on this analysis, we introduce a novel method, Hadamard-based Optimized Training (HOT). In this approach, we apply Hadamard-based optimizations, such as Hadamard quantization and Hadamard low-rank approximation, selectively and with awareness of the suitability of each optimization for different backward paths. Additionally, we introduce two enhancements: activation buffer compression and layer-wise quantizer selection. Our extensive analysis shows that HOT achieves up to 75% memory savings and a 2.6X acceleration on real GPUs, with negligible accuracy loss compared to FP32 precision.


### Training code
You can reproduce the accuracy, memory reduction, computation cost result of HOT during finetuning ViT-B with CIFAR100.

```
hot_cvpr2025/
├── main.sh    
├── memory_analysis.sh
├── gbops_analysis.sh
└── cuda_measure/
    └── measure.py
```

### How to run

- bash main.sh
- bash memory_analysis.sh
- bash gbops_analysis.sh

- latency comparison: 
    1. install CUTLASS
    2. set CUTLASS path in ./cuda_measure/setup_HLQ.py
    3. run "python setup_HLQ.py install"
    4. run "python measure.py"


### Cite
If you find our code or HOT useful for your research, please consider citing:
