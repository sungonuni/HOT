import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, DistributedSampler
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import os
import args

cifar10_stats = {'mean':[0.49139968, 0.48215827, 0.44653124],
                   'std': [0.2023, 0.1994, 0.2010]}

cifar100_stats = {'mean':[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                   'std': [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]}

def scale_crop(input_size, scale_size, normalize=cifar10_stats):
    t_list = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list
    if 'resnet' in args.MODEL:
        None
    elif 'swinv2' in args.MODEL:
        t_list.append(transforms.Resize(256, antialias=None))
    else:
        t_list.append(transforms.Resize(224, antialias=None))
    return transforms.Compose(t_list)

def pad_random_crop(input_size, scale_size, normalize=cifar10_stats):
    padding = int((scale_size - input_size) / 2)
    t_list = [
        transforms.RandomCrop(input_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if 'resnet' in args.MODEL:
        None
    elif 'swinv2' in args.MODEL:
        t_list.append(transforms.Resize(256, antialias=None))
    else:
        t_list.append(transforms.Resize(224, antialias=None))
    return transforms.Compose(t_list)

def gen_loaders(path, BATCH_SIZE, NUM_WORKERS, DATASET, DDP, GPU_NUM, input_size=None):
    # Data loading code
    path = os.path.join(path, DATASET)
    if DATASET == 'cifar10':
        train_dataset = datasets.CIFAR10(root=path,
                                        train=True,
                                        transform=pad_random_crop(input_size=32,
                                                                scale_size=40),
                                        download=True)

        val_dataset = datasets.CIFAR10(root=path,
                                        train=False,
                                        transform=scale_crop(input_size=32,
                                                            scale_size=32),
                                        download=True)
    elif DATASET == 'cifar100':
        train_dataset = datasets.CIFAR100(root=path,
                                        train=True,
                                        transform=pad_random_crop(input_size=32,
                                                                scale_size=40,
                                                                normalize=cifar100_stats),
                                        download=True)

        val_dataset = datasets.CIFAR100(root=path,
                                        train=False,
                                        transform=scale_crop(input_size=32,
                                                            scale_size=32,
                                                            normalize=cifar100_stats),
                                        download=True)
    elif DATASET == 'ImageNet100':
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        train_path = os.path.join(path, 'train')
        train_dataset = datasets.ImageFolder(train_path,
                                            transform=transforms.Compose([
                                            transforms.RandomResizedCrop(224 if input_size == None else input_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalize,
        ]))
        val_path = os.path.join(path, 'val')
        val_dataset = datasets.ImageFolder(val_path,
                                            transform=transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224 if input_size == None else input_size),
                                            transforms.ToTensor(),
                                            normalize,
        ]))
    
    elif DATASET == 'ILSVRC2012':
        
        normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                                     std=IMAGENET_DEFAULT_STD)
        train_path = os.path.join(path, 'train')
        train_dataset = datasets.ImageFolder(train_path,
                                            transform=transforms.Compose([
                                            transforms.RandomResizedCrop(224 if input_size == None else input_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalize,
        ]))
        val_path = os.path.join(path, 'val')
        val_dataset = datasets.ImageFolder(val_path,
                                            transform=transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224 if input_size == None else input_size),
                                            transforms.ToTensor(),
                                            normalize,
        ]))

    else:
        raise Exception('DATASET not implemented: ' + DATASET)

    if DDP == True:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        
        train_loader = DataLoader(train_dataset,
                                    sampler=train_sampler,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=NUM_WORKERS,
                                    pin_memory=True,
                                    drop_last=True)
        
        val_sampler = DistributedSampler(val_dataset, shuffle=True)

        val_loader = DataLoader(val_dataset,
                                sampler=val_sampler,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=True,
                                drop_last=True)
        
        return (train_loader, val_loader)
    else:
        train_loader = DataLoader(train_dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=NUM_WORKERS,
                                                pin_memory=True,
                                                drop_last=True)

        val_loader = DataLoader(val_dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=False,
                                                num_workers=NUM_WORKERS,
                                                pin_memory=True,
                                                drop_last=True)

        return (train_loader, val_loader)