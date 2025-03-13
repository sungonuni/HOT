import os
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import gc

import numpy as np

from utils.analysis_calculator import ModelMemoryCalculator

import random
import args
import argparse
import timm
from timm.models import create_model
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# Dataset class dict
dataset_class = {
    'cifar10' : 10,
    'cifar100' : 100,
    'ImageNet100' : 100,
    'ILSVRC2012': 1000,
}

quantScheme_gx = ["stoch_quantizer", "norm_quantizer"]
quantScheme_gw = ["stoch_quantizer", "norm_quantizer"]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('HLQ training and evaluation script', add_help=False)
    
    parser.add_argument('--GPU_USE', default='4', type=str)
    parser.add_argument('--RUN_NAME', default='', type=str)
    parser.add_argument('--DEBUG_MODE', type=str2bool, default=False, help='False when train')
    parser.add_argument('--MODEL', type=str, help='Q_efficientformer_l1, Q_efficientformerv2_l, Q_swinv2_b, Q_swinv2_l')
    parser.add_argument('--PRETRAINED', type=str2bool, help='imagenet pretrained')
    parser.add_argument('--CONTINUE', type=str2bool, default=False, help='continue training from ckpt file')
    parser.add_argument('--DATASET', type=str, help='cifar10, cifar100, ImageNet100, ILSVRC2012, voc')
    parser.add_argument('--AMP', type=str2bool, default=False, help='False when HLQ used')
    parser.add_argument('--EPOCHS', default=200, type=int)
    parser.add_argument('--BATCH_SIZE', default=128, type=int)
    parser.add_argument('--LR', default=0.1, type=float,
                        help='0.1 for resnet, 1e-3 for Eformer, 1e-1 for EformerV2, 0.256 for EfNet, 0.001 for EfNet_pt, 0.0003 for simclr, 5e-04 for swinv2, 5e-5 for segformer')
    parser.add_argument('--WORKERS', default=8, type=int)
    parser.add_argument('--DATA_DIR', type=str, default='/SSD')
    parser.add_argument('--CKPT_DIR', type=str, default='./checkpoint') 
    parser.add_argument('--SEED', default=2023, type=int)

    parser.add_argument('--LoRA', type=str2bool, default=False)
    parser.add_argument('--LoRA_all', type=str2bool, default=False)
    parser.add_argument('--HLQ_on_base', type=str2bool, default=False)
    parser.add_argument('--HLQ_on_decompose', type=str2bool, default=False)

    parser.add_argument('--precisionScheduling', type=str2bool, default=False, help='Enable for quant scheme is stoch or int')
    parser.add_argument('--milestone', default='50', type=str)
    parser.add_argument('--GogiQuantBit', default=4, type=int)
    parser.add_argument('--weightQuantBit', default=4, type=int)
    parser.add_argument('--GogwQuantBit', default=4, type=int)
    parser.add_argument('--actQuantBit', default=4, type=int)
    parser.add_argument('--eps', type=float, default=1e-11)

    parser.add_argument('--quantAuto', default=False, type=str2bool, help='Auto quant sheme')
    parser.add_argument('--quantBWDGogi', default='no', type=str, help='int, stoch, no, luq')
    parser.add_argument('--quantBWDWgt', default='no', type=str, help='int, stoch, no, luq')
    parser.add_argument('--quantBWDGogw', default='no', type=str, help='int, stoch, no, luq')
    parser.add_argument('--quantBWDAct', default='no', type=str, help='int, stoch, no, luq')

    parser.add_argument('--vectorPercentile', default=50, type=int)

    parser.add_argument('--transform_scheme', default='gih_gwlr', type=str, help='hadamard, low_rank, gih_gwlr(for matmul), gih_gwlrh(kernel), gilro_gwFP')

    parser.add_argument('--TransformGogi', type=str2bool, default=False)
    parser.add_argument('--TransformWgt', type=str2bool, default=False)
    parser.add_argument('--TransformGogw', type=str2bool, default=False)
    parser.add_argument('--TransformInput', type=str2bool, default=False)

    parser.add_argument('--DISTRIBUTED', type=str2bool, default=False)

    parser.add_argument('--wagSaveForPlot', type=str2bool, default=False)
    parser.add_argument('--wagSave_DIR', default='./pickle', type=str)
    parser.add_argument('--wagMilestone', default='0,50,100,150,199', type=str)
    
    return parser

def allocate_args(parsed_args):
    for name, _ in vars(parsed_args).items(): 
        setattr(args, name, getattr(parsed_args, name))

def main_worker(rank, parsed_args):    
    
    # allocate parsed_args to args
    allocate_args(parsed_args)
    
    if args.MODEL.endswith('resnet18'):
        model = timm.create_model('resnet18', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET])
    elif args.MODEL.endswith('resnet34'):
        model = timm.create_model('resnet34', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET])
    elif args.MODEL.endswith('resnet50'):
        model = timm.create_model('resnet50', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET])
    elif args.MODEL.endswith('resnet101'):
        model = timm.create_model('resnet101', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET])
    elif args.MODEL.endswith('resnet152'):
        model = timm.create_model('resnet152', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET])
    elif args.MODEL.endswith('efficientformer_l1'):
        model = create_model('efficientformer_l1', pretrained=parsed_args.PRETRAINED, distillation=None, num_classes=dataset_class[args.DATASET])
    elif args.MODEL.endswith('efficientformer_l3'):
        model = create_model('efficientformer_l3', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET])
    elif args.MODEL.endswith('efficientformer_l7'):
        model = create_model('efficientformer_l7', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET])
    elif args.MODEL.endswith('efficientformerv2_s0'):
        model = create_model('efficientformerv2_s0', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET])
    elif args.MODEL.endswith('efficientformerv2_l'):
        model = create_model('efficientformerv2_l', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET])
    elif args.MODEL.endswith('swinv2_b'):
        model = create_model('swinv2_base_window16_256', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET])
    elif args.MODEL.endswith('swinv2_l'):
        model = create_model('swinv2_large_window12to16_192to256', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET])
    elif args.MODEL.endswith('vit_s'):
        model = create_model('vit_small_patch16_224', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET])
    elif args.MODEL.endswith('vit_b'):
        model = create_model('vit_base_patch16_224', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET])
    else:
        raise NotImplementedError

    # LoRA
    if args.LoRA:
        target_module_list = ["qkv"]
    elif args.LoRA_all:
        target_module_list = ["fc1", "fc2", "qkv", "proj"]
    
    if args.LoRA or args.LoRA_all:
        lora_config = LoraConfig(
            r=8,                     
            lora_alpha=32,           
            target_modules=target_module_list,  
            lora_dropout=0.1,        
            bias="none",             
        )

        model = get_peft_model(model, lora_config)
    
    model.to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR, eps=1e-11)

    calculator = ModelMemoryCalculator(
        model=model,
        input_size=(3, 224, 224),
        batch_size=args.BATCH_SIZE,
        optimizer=optimizer,
        run_name=args.RUN_NAME
    )

    calculator.print_memory_usage()

    del calculator

    calculator = ModelMemoryCalculator(
        model=model,
        input_size=(3, 224, 224),
        batch_size=args.BATCH_SIZE,
        optimizer=optimizer,
        isHLQ=True,
        run_name=args.RUN_NAME+"-HLQ"
    )

    calculator.print_memory_usage()

    print("")
    print("")
    print("")

    del calculator

if __name__ == '__main__':
    # args parsing
    parser = argparse.ArgumentParser('HLQ training and evaluation script', parents=[get_args_parser()])
    parsed_args = parser.parse_args()
    
    # seed setting
    seed = parsed_args.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = parsed_args.GPU_USE
    parsed_args.GPU_NUM = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    parsed_args.DISTRIBUTED = False

    # Logging
    # print_current_opt(parsed_args)

    # Spawn multiprocess for each GPUs
    main_worker(0, parsed_args)