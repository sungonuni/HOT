import logging
import args
import os
import torch

def print_current_opt(args):
    logging.basicConfig(level=logging.INFO)
    logging.info('GPU: '+ str(os.environ["CUDA_VISIBLE_DEVICES"]))
    logging.info('RUN_NAME: '+ str(args.RUN_NAME))
    logging.info('DEBUG_MODE: '+ str(args.DEBUG_MODE))
    logging.info('MODEL: '+ str(args.MODEL))
    logging.info('PRETRAINED: '+ str(args.PRETRAINED))
    logging.info('CONTINUE: '+ str(args.CONTINUE))
    logging.info('DATASET: '+ str(args.DATASET))
    logging.info('AMP: '+ str(args.AMP))
    logging.info('EPOCHS: '+ str(args.EPOCHS))
    logging.info('BATCH_SIZE: '+ str(args.BATCH_SIZE))
    logging.info('Learning Rate: '+ str(args.LR))
    logging.info('LoRA: '+ str(args.LoRA))
    logging.info('LoRA_all: '+ str(args.LoRA_all))
    logging.info('HLQ_on_base: '+ str(args.HLQ_on_base))
    logging.info('HLQ_on_decompose: '+ str(args.HLQ_on_decompose))
    logging.info('WORKERS: '+ str(args.WORKERS))
    logging.info('DATA_DIR: '+ str(args.DATA_DIR))
    logging.info('CKPT_DIR: '+ str(args.CKPT_DIR))
    logging.info('SEED: '+ str(args.SEED))
    logging.info('precisionScheduling: '+ str(args.precisionScheduling))
    logging.info('milestone: '+ str(args.milestone))
    logging.info('GogiQuantBit: '+ str(args.GogiQuantBit))
    logging.info('weightQuantBit: '+ str(args.weightQuantBit))
    logging.info('GogwQuantBit: '+ str(args.GogwQuantBit))
    logging.info('actQuantBit: '+ str(args.actQuantBit))
    logging.info('eps: '+ str(args.eps))
    logging.info('quantBWDGogi: '+ str(args.quantBWDGogi))
    logging.info('quantBWDWgt: '+ str(args.quantBWDWgt))
    logging.info('quantBWDGogw: '+ str(args.quantBWDGogw))
    logging.info('quantBWDAct: '+ str(args.quantBWDAct))
    logging.info('vectorPercentile: '+ str(args.vectorPercentile))
    logging.info('transform_scheme: '+ str(args.transform_scheme))
    logging.info('TransformGogi: '+ str(args.TransformGogi))
    logging.info('TransformWgt: '+ str(args.TransformWgt))
    logging.info('TransformGogw: '+ str(args.TransformGogw))
    logging.info('TransformInput: '+ str(args.TransformInput))
    logging.info('wagSaveForPlot: '+ str(args.wagSaveForPlot))
    logging.info('wagSave_DIR: '+ str(args.wagSave_DIR))
    logging.info('wagMilestone: '+ str(args.wagMilestone))


def create_checkpoint(model, optimizer, lr_scheduler, epoch, ckpt_dir, run_name):
    # Create checkpoint dir
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    
    # Set filename
    # filename = run_name + '_{}_{}.ckpt'.format(prefix, epoch)
    filename = run_name + '.ckpt'
    filepath = os.path.join(ckpt_dir, filename)

    # Check instansce
    if isinstance(model, torch.nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch
    }, filepath)    


def categoraize_param(model):
    weight = []
    quant = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif name.endswith("sw"):
            quant.append(param)
        else:
            weight.append(param)
    return (weight, quant)
