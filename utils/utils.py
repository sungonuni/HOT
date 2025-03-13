import math
from torch.optim.lr_scheduler import _LRScheduler
import args
import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_z_score_threshold(data, threshold=2.5):
    # data shape: (batch, length)
    mean = torch.mean(data, dim=1, keepdim=True)  # (batch, 1)
    std = torch.std(data, dim=1, keepdim=True)    # (batch, 1)
    return mean + threshold * std           # (batch, 1)

def batched_fast_otsu_threshold(tensor, num_bins=256):
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2-dimensional (batch, length)")

    # Compute the min and max of the data for each batch
    data_min = tensor.min(dim=1)[0]
    data_max = tensor.max(dim=1)[0]
    
    # Compute histogram and probabilities for each batch
    batch_size = tensor.shape[0]
    hist = torch.stack([torch.histc(tensor[i], bins=num_bins, min=data_min[i], max=data_max[i]) 
                        for i in range(batch_size)])
    hist_norm = hist / hist.sum(dim=1, keepdim=True)

    # Compute bin centers for each batch
    bin_edges = torch.stack([torch.linspace(data_min[i], data_max[i], num_bins + 1) 
                             for i in range(batch_size)]).to(tensor.device)
    bin_centers = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2

    # Compute cumulative sums
    cum_sum = torch.cumsum(hist_norm, dim=1)
    cum_mean = torch.cumsum(bin_centers * hist_norm, dim=1)

    # Compute global mean
    global_mean = cum_mean[:, -1]

    # Compute between-class variance
    bcv = (global_mean.unsqueeze(1) * cum_sum - cum_mean) ** 2 / (cum_sum * (1 - cum_sum + 1e-10))

    # Find the threshold that maximizes the between-class variance
    _, max_idx = torch.max(bcv[:, 1:-1], dim=1)  # Exclude first and last
    thresholds = torch.stack([bin_edges[i, idx + 1] for i, idx in enumerate(max_idx)])

    return thresholds

'''
Replace the nn.Linear and nn.Conv2d to HQ_Linear, HQ_Conv2d
'''
def replace_layers(model, old, new):
    def replace(model, new_layer, module):
        new_layer.weight.data = module.weight.data
        new_layer.weight.requires_grad = module.weight.requires_grad
        if module.bias is not None:
            new_layer.bias.data = module.bias.data
        setattr(model, n, new_layer)

    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(module, old, new)
            
        if old == nn.Conv2d and isinstance(module, old) and module.groups == 1: # not for dw
            new_layer = new(module.in_channels, module.out_channels, kernel_size=module.kernel_size, stride=module.stride, padding=module.padding)
            replace(model, new_layer, module)
        elif old == nn.Linear and isinstance(module, old):
            if 'base_layer' in n and args.HLQ_on_base == False:
                continue
            if 'default' in n and args.HLQ_on_decompose == False:
                continue
            new_layer = new(module.in_features, module.out_features)
            replace(model, new_layer, module)


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def precision_scheduler(current_epoch, max_bit=8, min_bit=4, milestones=[]):
    if current_epoch == 0:
        args.GogiQuantBit = max_bit
        args.weightQuantBit = max_bit
        args.GogwQuantBit = max_bit
        args.actQuantBit = max_bit
    elif current_epoch in milestones and min(args.GogiQuantBit, args.weightQuantBit, args.GogwQuantBit, args.actQuantBit) > min_bit:
        if args.quantBWDGogi == "stoch" or args.quantBWDGogi == "int":
            args.GogiQuantBit = args.GogiQuantBit // 2
        if args.quantBWDWgt == "stoch" or args.quantBWDWgt == "int":
            args.weightQuantBit = args.weightQuantBit // 2

def dw_unfold(input, kernel_size, groups, dilation=1, padding=(1,1), stride=(1,1)):
    _,C,_,_ = input.shape
    assert groups == C
    input_splits = input.split(1, dim=1)
    input_splits_unf = []
    for i in range(C):
        input_unf = F.unfold(input_splits[i], kernel_size, dilation=dilation, padding=padding, stride=stride).transpose(1, 2)
        input_splits_unf.append(input_unf)
    batched_block_diag = torch.vmap(torch.block_diag)
    return batched_block_diag(*[input_splits_unf[i] for i in range(C)])

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(
                outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss

"""
Under functions are only for EfficientNetV2 and Q_EfficientNetV2!!
"""

def weight_decay(model, decay=1e-5):
    p1 = []
    p2 = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            p1.append(param)
        else:
            p2.append(param)
    return [{'params': p1, 'weight_decay': 0.}, {'params': p2, 'weight_decay': decay}]

class StepLR:
    def __init__(self, optimizer):
        self.optimizer = optimizer

        for param_group in self.optimizer.param_groups:
            param_group.setdefault('initial_lr', param_group['lr'])

        self.base_values = [param_group['initial_lr'] for param_group in self.optimizer.param_groups]
        self.update_groups(self.base_values)

        self.decay_rate = 0.97
        self.decay_epochs = 2.4
        self.warmup_epochs = 3.0
        self.warmup_lr_init = 1e-6

        self.warmup_steps = [(v - self.warmup_lr_init) / self.warmup_epochs for v in self.base_values]
        self.update_groups(self.warmup_lr_init)

    def step(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            values = [self.warmup_lr_init + epoch * s for s in self.warmup_steps]
        else:
            values = [v * (self.decay_rate ** (epoch // self.decay_epochs)) for v in self.base_values]
        if values is not None:
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value


class RMSprop(torch.optim.Optimizer):
    def __init__(self, params,
                 lr=1e-2, alpha=0.9, eps=1e-3, weight_decay=0, momentum=0.9,
                 centered=False, decoupled_decay=False, lr_in_momentum=True):

        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum,
                        centered=centered, decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for param_group in self.param_groups:
            param_group.setdefault('momentum', 0)
            param_group.setdefault('centered', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Optimizer does not support sparse gradients')
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(param.data)
                    if param_group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(param.data)
                    if param_group['centered']:
                        state['grad_avg'] = torch.zeros_like(param.data)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - param_group['alpha']

                state['step'] += 1

                if param_group['weight_decay'] != 0:
                    if 'decoupled_decay' in param_group and param_group['decoupled_decay']:
                        param.data.add_(param.data, alpha=-param_group['weight_decay'])
                    else:
                        grad = grad.add(param.data, alpha=param_group['weight_decay'])

                square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)

                if param_group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add(param_group['eps']).sqrt_()
                else:
                    avg = square_avg.add(param_group['eps']).sqrt_()

                if param_group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    if 'lr_in_momentum' in param_group and param_group['lr_in_momentum']:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg, value=param_group['lr'])
                        param.data.add_(-buf)
                    else:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg)
                        param.data.add_(-param_group['lr'], buf)
                else:
                    param.data.addcdiv_(grad, avg, value=-param_group['lr'])

        return loss


def info_nce_loss(features, batch_size, n_views): # Only for simclr

        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to('cuda')

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to('cuda')
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to('cuda')

        logits = logits / 0.07 # softmax temperature
        return logits, labels