from collections import defaultdict
from utils.plot import Tensor_Dict2

# Will be deprecated
GPU_USE = " "
RUN_NAME = ' '
DEBUG_MODE = False # False when train
MODEL = ' ' # Q_resnet18, Q_resnet34, Q_efficientnetv2_s, Q_resnet50, Q_efficientformer_l1, Q_efficientformerv2_l, Q_swinv2_b, Q_swinv2_l 
PRETRAINED = False
CONTINUE = False
DATASET = ' ' # cifar10, cifar100, ImageNet100, ILSVRC2012, voc
AMP = False # False when HLQ used
EPOCHS = 0
BATCH_SIZE = 0 # simclr 256, others 128, segformer 16 
LR = 0 # 0.1 for resnet, 1e-3 for Eformer, 1e-1 for EformerV2, 0.256 for EfNet, 0.001 for EfNet_pt, 0.0003 for simclr, 5e-04 for swinv2, 5e-5 for segformer

LoRA = False
LoRA_all = False
HLQ_on_base = False
HLQ_on_decompose = False

WORKERS = 0
DATA_DIR = ' '
CKPT_DIR = ' '
SEED = 0

precisionScheduling = False # Enable for quant scheme is stoch or int
milestone = " "
GogiQuantBit = 0
weightQuantBit = 0
GogwQuantBit = 0
actQuantBit = 0

eps = 1e-11

quantAuto = False
quantBWDGogi = "no" # int, stoch, no, luq
quantBWDWgt = "no" # int, stoch, no, sawb
quantBWDGogw = "no" # int, stoch, no, luq
quantBWDAct = "no" # int, stoch, no, sawb

vectorPercentile = 50

transform_scheme = "gih_gwlr" # hadamard, low_rank, gih_gwlr(for matmul), gih_gwlrh(kernel), gilro_gwFP

TransformGogi = False
TransformWgt = False
TransformGogw = False
TransformInput = False

KERNEL = False
DISTRIBUTED = False

wagSaveForPlot = False # Currently available only for resnet and EffFormer
wagSave_DIR = ' '
wagMilestone = " "

layer_quant_dict = defaultdict()
captured_tensor_dict = Tensor_Dict2()
