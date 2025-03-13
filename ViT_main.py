import os
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import gc

import numpy as np
from tqdm import tqdm
import time
from utils.data import gen_loaders
from utils.meters import AverageMeter

from utils.utils import precision_scheduler, DistillationLoss, replace_layers
from utils.plot import Tensor_Dict
from utils.logging import create_checkpoint
from utils.quant_matmul import HQ_Conv2d, HQ_Linear, stoch_quantizer, norm_quantizer

import random
import args
import argparse
from timm.utils import accuracy
from timm.data import Mixup
from timm.models import create_model
from timm.loss import SoftTargetCrossEntropy
from torch.cuda.amp import GradScaler, autocast
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

tensor_dict = Tensor_Dict()

def forward_hook(module, input, output):
    tensor_dict.register('weight', module.weight, direction='forward')
    tensor_dict.register('input', input[0], direction='forward')

def backward_hook(module, grad_input, grad_output):
    tensor_dict.register('grad_out', grad_output[0], direction='backward')

def layer_wag_save_init(model):
    for name, layer in model.named_modules():
        if isinstance(layer, HQ_Conv2d) or isinstance(layer, HQ_Linear):
            tensor_dict.key_init(name)
            layer.register_forward_hook(forward_hook)
            layer.register_full_backward_hook(backward_hook)

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
    parser.add_argument('--LR', default=0.1, type=float)
    parser.add_argument('--WORKERS', default=8, type=int)
    parser.add_argument('--DATA_DIR', type=str, default='/SSD') # Change this to train Imagenet dataset.
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

    parser.add_argument('--quantAuto', default=True, type=str2bool, help='Auto quant sheme')
    parser.add_argument('--quantBWDGogi', default='nq', type=str, help='int, stoch, no, luq')
    parser.add_argument('--quantBWDWgt', default='sawb', type=str, help='int, stoch, no, luq')
    parser.add_argument('--quantBWDGogw', default='nq', type=str, help='int, stoch, no, luq')
    parser.add_argument('--quantBWDAct', default='sawb', type=str, help='int, stoch, no, luq')

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

def allocate_args_for_calibration(parsed_args):
    for name, _ in vars(parsed_args).items():
        if name.startswith('quant') or name.startswith('Transform'):
            continue
        setattr(args, name, getattr(parsed_args, name))

def calibaration_worker(rank, parsed_args):
    allocate_args_for_calibration(parsed_args)

    if parsed_args.MODEL.endswith('efficientformer_l1'):
        model = create_model('efficientformer_l1', pretrained=parsed_args.PRETRAINED, distillation=None, num_classes=dataset_class[parsed_args.DATASET]); model_ema = None
    elif parsed_args.MODEL.endswith('efficientformer_l3'):
        model = create_model('efficientformer_l3', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[parsed_args.DATASET]); model_ema = None
    elif parsed_args.MODEL.endswith('efficientformer_l7'):
        model = create_model('efficientformer_l7', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[parsed_args.DATASET]); model_ema = None
    elif parsed_args.MODEL.endswith('efficientformerv2_s0'):
        model = create_model('efficientformerv2_s0', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[parsed_args.DATASET]); model_ema = None
    elif parsed_args.MODEL.endswith('efficientformerv2_l'):
        model = create_model('efficientformerv2_l', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[parsed_args.DATASET]); model_ema = None
    elif parsed_args.MODEL.endswith('swinv2_b'):
        model = create_model('swinv2_base_window16_256', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[parsed_args.DATASET]); model_ema = None
    elif parsed_args.MODEL.endswith('swinv2_l'):
        model = create_model('swinv2_large_window12to16_192to256', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[parsed_args.DATASET]); model_ema = None
    elif parsed_args.MODEL.endswith('vit_s'):
        model = create_model('vit_small_patch16_224', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[parsed_args.DATASET]); model_ema = None
    elif parsed_args.MODEL.endswith('vit_b'):
        model = create_model('vit_base_patch16_224', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[parsed_args.DATASET]); model_ema = None
    else:
        raise NotImplementedError
    
    if parsed_args.MODEL.startswith('Q'):
        replace_layers(model, nn.Conv2d, HQ_Conv2d)
        replace_layers(model, nn.Linear, HQ_Linear)
        for name, layer in model.named_modules():
            if isinstance(layer, HQ_Conv2d) or isinstance(layer, HQ_Linear):
                layer.layer_name = name

    model.to('cuda')
    
    layer_wag_save_init(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR, eps=1e-11)
    criterion = torch.nn.CrossEntropyLoss()
    _, test_loader = gen_loaders(parsed_args.DATA_DIR, 32, parsed_args.WORKERS, parsed_args.DATASET, DDP=False, GPU_NUM=1, input_size=None)

    print("=== Layer-wise Quantizer Selection (LQS) ===")
    for i, data in enumerate(tqdm(test_loader)):
        inputs, target = data
        inputs = inputs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    for name in tensor_dict.module_dict.keys():
        original_grad = tensor_dict.module_dict[name]['grad_out']
        if parsed_args.MODEL.endswith('efficientformer_l1') or parsed_args.MODEL.endswith('efficientformer_l3') or parsed_args.MODEL.endswith('efficientformer_l7'):
            if len(original_grad.shape) > 3:
                B,C,H,W = original_grad.shape
                original_grad = original_grad.permute(0, 2, 3, 1).reshape(B, H*W, C)

        L = original_grad.shape[1] if len(original_grad.shape) > 2 else original_grad.shape[0]
        O = tensor_dict.module_dict[name]['weight'].shape[0]
        I = tensor_dict.module_dict[name]['weight'].shape[1]
        is_batched = True if len(original_grad.shape) == 3 else False
        min_qerror = 9999
        for q_func in quantScheme_gx:
            if q_func == "stoch_quantizer":
                q_grad, s_grad = stoch_quantizer(original_grad, 4, is_batched)
            elif q_func == "norm_quantizer":
                q_grad, s_grad = norm_quantizer(original_grad, 4, is_batched)
            else:
                raise NotImplementedError 
            qerror = torch.dist(original_grad, q_grad * (s_grad.unsqueeze(1).unsqueeze(2) if is_batched and torch.is_tensor(s_grad) else s_grad))
            if qerror < min_qerror and q_func != "norm_quantizer":
                min_qerror = qerror
                args.layer_quant_dict[name+'_gx'] = q_func
                args.layer_quant_dict[name+'_gw'] = q_func
            elif qerror < min_qerror and q_func == "norm_quantizer":
                if (min_qerror - qerror) > (min_qerror*0.5):
                    min_qerror = qerror
                    args.layer_quant_dict[name+'_gx'] = q_func
                    args.layer_quant_dict[name+'_gw'] = "stoch_quantizer"

    tensor_dict.value_clear()
    del model, optimizer, criterion, _, test_loader
    torch.cuda.empty_cache()

def main_worker(rank, parsed_args):    
    
    # allocate parsed_args to args
    allocate_args(parsed_args)
    GPU_NUM = parsed_args.GPU_NUM

    if args.DISTRIBUTED:
        # DDP setup
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:1111'+str(args.GPU_USE[0]), world_size=GPU_NUM, rank=rank)

    # mixup fn
    mixup_fn = Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=dataset_class[args.DATASET])

    if args.MODEL.endswith('efficientformer_l1'):
        model = create_model('efficientformer_l1', pretrained=parsed_args.PRETRAINED, distillation=None, num_classes=dataset_class[args.DATASET]); model_ema = None
    elif args.MODEL.endswith('efficientformer_l3'):
        model = create_model('efficientformer_l3', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET]); model_ema = None
    elif args.MODEL.endswith('efficientformer_l7'):
        model = create_model('efficientformer_l7', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET]); model_ema = None
    elif args.MODEL.endswith('efficientformerv2_s0'):
        model = create_model('efficientformerv2_s0', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET]); model_ema = None
    elif args.MODEL.endswith('efficientformerv2_l'):
        model = create_model('efficientformerv2_l', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET]); model_ema = None
    elif args.MODEL.endswith('swinv2_b'):
        model = create_model('swinv2_base_window16_256', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET]); model_ema = None
    elif args.MODEL.endswith('swinv2_l'):
        model = create_model('swinv2_large_window12to16_192to256', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET]); model_ema = None
    elif args.MODEL.endswith('vit_s'):
        model = create_model('vit_small_patch16_224', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET]); model_ema = None
    elif args.MODEL.endswith('vit_b'):
        model = create_model('vit_base_patch16_224', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET]); model_ema = None
    else:
        raise NotImplementedError

    # LoRA
    if args.LoRA:
        target_module_list = ["qkv"]
    elif args.LoRA_all:
        target_module_list = ["fc1", "fc2", "qkv", "proj"]
    
    if args.LoRA or args.LoRA_all:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_module_list,
            lora_dropout=0.1,
            bias="none",
        )

        model = get_peft_model(model, lora_config)

    if args.MODEL.startswith('Q'):
        replace_layers(model, nn.Conv2d, HQ_Conv2d)
        replace_layers(model, nn.Linear, HQ_Linear)
        for name, layer in model.named_modules():
            if isinstance(layer, HQ_Conv2d) or isinstance(layer, HQ_Linear):
                layer.layer_name = name
    
    model.to('cuda')

    teacher_model = None
    
    # DDP sync
    if args.DISTRIBUTED:
        model = DDP(model, device_ids=[rank], output_device=0, find_unused_parameters=True)
        dist.barrier()
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR, eps=1e-11)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    if args.CONTINUE:
        ckpt_name = args.RUN_NAME + ".ckpt"
        ckpt = torch.load(os.path.join(args.CKPT_DIR, ckpt_name))
        if not args.DISTRIBUTED:
            for key in list(ckpt['model'].keys()):
                if 'module.' in key:
                    ckpt['model'][key.replace('module.', '')] = ckpt['model'][key]
                    del ckpt['model'][key]
        
        for key in list(ckpt['model'].keys()):
            if 'hadamard_bs' in key:
                del ckpt['model'][key]
        
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['lr_scheduler'])

        del ckpt
        gc.collect()
        torch.cuda.empty_cache()
    
    criterion = SoftTargetCrossEntropy().to('cuda')
    criterion = DistillationLoss(
        criterion, teacher_model, distillation_type='none', alpha=0.5, tau=1.0,
    )
    valid_criterion = torch.nn.CrossEntropyLoss()

    train_loader, test_loader = gen_loaders(args.DATA_DIR, args.BATCH_SIZE, args.WORKERS, args.DATASET, DDP=(True if GPU_NUM > 1 else False), GPU_NUM=GPU_NUM, input_size=256 if 'swinv2' in args.MODEL else None)

    if args.wagSaveForPlot:
        layer_wag_save_init(model)

    scaler = GradScaler(enabled=True)
    best_acc = 0
    for epoch in range(args.EPOCHS):
        
        if args.precisionScheduling:
            precision_scheduler(current_epoch=epoch, max_bit=8, min_bit=4, milestones=list(map(int, args.milestone.split(','))))

        train_loss, train_prec1, train_prec5= forward(
            epoch, scaler, train_loader, model, model_ema, criterion, optimizer, training=True, mixup_fn=mixup_fn)
        
        if rank == 0:
            with torch.no_grad():
                val_loss, val_prec1, val_prec5= forward(
                    epoch, scaler, test_loader, model, model_ema, valid_criterion, optimizer, training=False, mixup_fn=None)
            
            scheduler.step()

            print('Epoch: {0} '
                        'Train Prec@1 {train_prec1:.3f} '
                        'Train Loss {train_loss:.3f} '
                        'Valid Prec@1 {val_prec1:.3f} '
                        'Valid Loss {val_loss:.3f} \n'
                        .format(epoch,
                                train_prec1=train_prec1, val_prec1=val_prec1,
                                train_loss=train_loss, val_loss=val_loss))

            if val_prec1 > best_acc and not args.DEBUG_MODE:
                best_acc = max(val_prec1, best_acc)
                create_checkpoint(model=model, optimizer=optimizer, lr_scheduler=scheduler, epoch=epoch, ckpt_dir=args.CKPT_DIR, run_name=args.RUN_NAME)
            
            if args.wagSaveForPlot:
                tensor_dict.pickle_save(epoch=epoch, wagsave_dir=args.wagSave_DIR, run_name=args.RUN_NAME)
                tensor_dict.value_clear()


def forward(epoch, scaler, data_loader, model, model_ema, criterion, optimizer, training, mixup_fn=None):
    if training:
        model.train()
    else:
        model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    for i, data in enumerate(tqdm(data_loader)):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, target = data
        inputs = inputs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        original_target = target
        if mixup_fn is not None:
            inputs, target = mixup_fn(inputs, target)

        optimizer.zero_grad()
        with autocast(enabled=args.AMP):
            output = model(inputs)
            if training:
                loss = criterion(inputs, output, target)
            else:
                loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss, inputs.size(0))
        prec1, prec5 = accuracy(output, original_target, topk=(1, 5))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        if training and args.AMP:            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if model_ema is not None:
                model_ema.update(model)
        elif training and not args.AMP:
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, top1.avg, top5.avg

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
    parsed_args.DISTRIBUTED = True if parsed_args.GPU_NUM > 1 else False

    # Get quant sheme per layer by calibration set
    if parsed_args.MODEL.startswith('Q') and parsed_args.quantAuto:
        calibaration_worker(0, parsed_args)

    # Spawn multiprocess for each GPUs
    if parsed_args.DISTRIBUTED:
        try: 
            mp.spawn(main_worker, args=(parsed_args), nprocs=parsed_args.GPU_NUM)
        except KeyboardInterrupt:
            print("Keyboard interrupted")
            dist.destroy_process_group()
    else:
        main_worker(0, parsed_args)
