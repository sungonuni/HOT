import torch
import torch.nn as nn
from typing import Dict, Tuple
import numpy as np
from typing import Dict, Tuple, Optional, Any
import torch.optim as optim
from collections import defaultdict
import math


class ModelMemoryCalculator:
    def __init__(
        self, 
        model: nn.Module, 
        input_size: Tuple, 
        batch_size: int, 
        optimizer: Optional[torch.optim.Optimizer] = None,
        isHLQ: bool = False,
        run_name: str = '',
        dtype: torch.dtype = torch.float32
    ):
        self.model = model
        self.input_size = input_size
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.bytes_per_element = torch.finfo(dtype).bits // 8
        self.isHLQ = isHLQ
        self.run_name = run_name

    def get_parameter_memory(self) -> Dict[str, float]:
        total_params = 0
        param_memory = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'qkv' in name:
                    num_params = param.numel() * 3
                else:
                    num_params = param.numel()
                memory = num_params * self.bytes_per_element
                param_memory[name] = memory
                total_params += num_params
            
        param_memory['total'] = total_params * self.bytes_per_element
        return param_memory
    
    def get_activation_memory(self) -> Dict[str, float]:
        activation_memory = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if self.isHLQ:
                    activation_memory[name] = ((output.numel() // 2) * self.batch_size) if module.weight.requires_grad else 0
                else:
                    activation_memory[name] = (output.numel() * self.batch_size * self.bytes_per_element) if module.weight.requires_grad else 0
                if 'qkv' in name:
                    activation_memory[name] *= 4

            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
                
        dummy_input = torch.randn(1, *self.input_size).to('cuda')
        self.model(dummy_input)
        
        for hook in hooks:
            hook.remove()
            
        return activation_memory
    
    def get_gradient_memory(self) -> float:
        param_memory = self.get_parameter_memory()
        return param_memory['total']
    
    def get_optimizer_state_memory(self) -> Dict[str, float]:
        if self.optimizer is None:
            return {'total': 0}
            
        optimizer_memory = {}
        total_memory = 0
        
        if isinstance(self.optimizer, optim.AdamW):
            param_memory = self.get_parameter_memory()
            total_memory = 2 * param_memory['total']
            
        optimizer_memory['total'] = total_memory
        return optimizer_memory
    
    def get_total_memory(self) -> Dict[str, float]:
        param_memory = self.get_parameter_memory()
        activation_memory = self.get_activation_memory()
        gradient_memory = self.get_gradient_memory()
        optimizer_memory = self.get_optimizer_state_memory()
        
        # import pdb; pdb.set_trace()
        total_memory = {
            'parameters': param_memory['total'],
            'activations': sum(activation_memory.values()),
            'gradients': gradient_memory,
            'optimizer_state': optimizer_memory['total'],
            'total': param_memory['total'] + 
                    sum(activation_memory.values()) + 
                    gradient_memory + 
                    optimizer_memory['total']
        }
        
        return total_memory
    
    def format_bytes(self, bytes: float) -> str:
        # if bytes < 1024:
        #     return f"{bytes:.2f}B"
        # elif bytes < 1024**2:
        #     return f"{bytes/1024:.2f}KB"
        # elif bytes < 1024**3:
        #     return f"{bytes/1024**2:.2f}MB"
        # else:
        #     return f"{bytes/1024**3:.2f}GB"
        return f"{bytes/1024**3:.2f}GB"
    
    def print_memory_usage(self):
        memory_usage = self.get_total_memory()
        
        print(f"===== Analysis of Memory footprint: {self.run_name}-{self.batch_size}=====")
        print(f"Weight: {self.format_bytes(memory_usage['parameters'])}")
        print(f"Activation: {self.format_bytes(memory_usage['activations'])}")
        print(f"Gradient: {self.format_bytes(memory_usage['gradients'])}")
        print(f"Optimizer State: {self.format_bytes(memory_usage['optimizer_state'])}")
        print(f"Total: {self.format_bytes(memory_usage['total'])}")
        print("=========================================")

class BOPsCounter:
    def __init__(self, transform_scheme):
        self.forward_gbops = defaultdict(int)
        self.backward_gbops = defaultdict(int)
        self.handles = []
        self.transform_scheme = transform_scheme
    
    def calculate_backword_bops(self, transform_scheme, param_pack, name):
        L, O, I, r, n = param_pack

        if transform_scheme == "FP" or transform_scheme == "LoRA":
            backward_gbops = L * O * I * (32*32) * 2
        elif transform_scheme == "LBP":
            backward_gbops = (L*O*np.log2(n) + L*I*np.log2(n) + O*I*L*(r/n))*(32*32)

        elif transform_scheme == "LUQ":
            backward_gbops = ((L*O + O*I + L*I + O*I + L*I)*(32*32)*3 # Logarithm Quant
                              ) + L*O*I*(4*4*2) + L*O*I*(4*4*2) # FP4 matmul and INT4 matmul
        elif transform_scheme == "HLQ":
            if 'base_layer' in name: # Freezed layer in LoRA
                backward_gbops = (L*O*np.log2(n) + O*I*np.log2(n) + L*O + O*I # gx path HT+Q\
                                        + L*I # dequent
                                        )* (32*32) + L*O*I*(4*4) # INT MatMul
            else:
                backward_gbops = (L*O*np.log2(n) + O*I*np.log2(n) + L*O + O*I # gx path HT+Q
                                        + L*I*np.log2(n) + L*O*np.log2(n) + O*(r/n)*L + I*(r/n)*O # gw path HLA+Q
                                        + I*O + L*I # dequent
                                        )* (32*32) + O*(r/n)*L*I*(8*8) + L*O*I*(4*4) # INT MatMul
        else:
            raise NotImplementedError
        
        return backward_gbops

    def compute_conv2d_gbops(self, module: nn.Conv2d, input_shape: torch.Size, output_shape: torch.Size, name: str) -> tuple:
        out_h = output_shape[-1]
        out_w = output_shape[-2]

        L = out_h * out_w
        O = module.out_channels
        I = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
        r = 8 # low-rank
        n = 16 # Hadamard size

        param_pack = (L, O, I, r, n)
        
        forward_gbops = L * O * I * (32*32)

        backward_gbops = self.calculate_backword_bops(self.transform_scheme, param_pack, name)
        
        return forward_gbops, backward_gbops
    
    def compute_linear_gbops(self, module: nn.Linear, input_shape: torch.Size, name: str) -> tuple:
        
        L = input_shape[1] if len(input_shape) == 3 else input_shape[0]
        O = module.out_features
        I = module.in_features
        r = 8 # low-rank
        n = 16 # Hadamard size

        param_pack = (L, O, I, r, n)

        forward_gbops = L * O * I * (32*32)
        backward_gbops = self.calculate_backword_bops(self.transform_scheme, param_pack, name)
        
        return forward_gbops, backward_gbops
    
    def hook_fn(self, name):
        def hook(module, input, output):
            input_shape = input[0].shape
            output_shape = output.shape
            if isinstance(module, nn.Conv2d):
                forward_gbops, backward_gbops = self.compute_conv2d_gbops(module, input_shape, output_shape, name)
            elif isinstance(module, nn.Linear):
                forward_gbops, backward_gbops = self.compute_linear_gbops(module, input_shape, name)
            else:
                return
            
            self.forward_gbops[name] = forward_gbops
            self.backward_gbops[name] = backward_gbops

        return hook
    
    def start_counting(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handle = module.register_forward_hook(self.hook_fn(name))
                self.handles.append(handle)
    
    def stop_counting(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def get_total_gbops(self) -> Dict[str, Any]:
        total_forward = sum(self.forward_gbops.values())
        total_backward = sum(self.backward_gbops.values())
        
        return {
            'total_forward_gbops': total_forward,
            'total_backward_gbops': total_backward,
            'total_gbops': total_forward + total_backward,
            'forward_gbops_by_layer': dict(self.forward_gbops),
            'backward_gbops_by_layer': dict(self.backward_gbops)
        }

    def reset(self) -> None:
        self.forward_gbops.clear()
        self.backward_gbops.clear()