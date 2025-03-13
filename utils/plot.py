import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import matplotlib.pyplot as plt
import pickle
from decimal import Decimal
import numpy as np

from collections import defaultdict
from functools import partial
from matplotlib.ticker import ScalarFormatter, LinearLocator
from matplotlib.ticker import MaxNLocator

class XYPlot():
    def __init__(self):
        super(XYPlot, self).__init__()
        self.accum_a = torch.Tensor(0).to('cpu')

        plt.rc('font', size=11)        
        plt.rc('axes', labelsize=11)   
        plt.rc('xtick', labelsize=11)  
        plt.rc('ytick', labelsize=11)  
        plt.rc('legend', fontsize=11)  
        plt.rc('figure', titlesize=12)
    
    def accumulate_a(self, input):
        reshape_input = input[0].abs().float().to('cpu') if len(input.shape) > 2 else input.abs().float().to('cpu')
        self.accum_a = torch.concat((self.accum_a, reshape_input), 0)

    def clear_a(self):
        self.accum_a = torch.Tensor(0).to('cpu')
    
    def get_a(self):
        return self.accum_a

    def draw_plot(self, run_name, png_name, lim=None, threshold=None, qerror=None):
        pngsave_dir = os.path.join('./png',run_name)
        if not os.path.exists(pngsave_dir):
            os.mkdir(pngsave_dir)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel('token')
        ax.set_ylabel('variance')
        if lim is not None:
            ax.set_ylim(bottom=0, top=lim)
        vars = torch.var(self.accum_a, dim=1)
        max_val = torch.max(self.accum_a, dim=1)
        colors = 'purple'
        if threshold is not None:
            colors = ['blue' if v <= threshold else 'red' for v in max_val[0]]

        plt.bar(range(self.accum_a.shape[0]), vars, color=colors)
        plt.grid(True, axis='y')
        plt.title(png_name)
        plt.savefig(os.path.join(pngsave_dir,(png_name+'.png')))
        plt.cla()
        plt.clf()
        plt.close()


class XYZplot():
    def __init__(self):
        super(XYZplot, self).__init__()
        self.accum_a = torch.Tensor(0).to('cpu')

        plt.rc('font', size=12)        
        plt.rc('axes', labelsize=12)   
        plt.rc('xtick', labelsize=12)  
        plt.rc('ytick', labelsize=12) 
        plt.rc('legend', fontsize=12) 
        plt.rc('figure', titlesize=14)
    
    def accumulate_a(self, input):
        reshape_input = input[0].abs().float().to('cpu') if len(input.shape) > 2 else input.abs().float().to('cpu')
        self.accum_a = torch.concat((self.accum_a, reshape_input), 0)

    def clear_a(self):
        self.accum_a = torch.Tensor(0).to('cpu')
    
    def get_a(self):
        return self.accum_a

    def draw_plot(self, run_name, png_name, lim=None, qerror=None):
        pngsave_dir = os.path.join('./png',run_name)
        if not os.path.exists(pngsave_dir):
            os.mkdir(pngsave_dir)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_range = torch.arange(self.accum_a.shape[0]) # token
        y_range = torch.arange(self.accum_a.shape[1]) # channel
        X, Y = torch.meshgrid(x_range, y_range)
        im = ax.plot_surface(X.detach().numpy(), Y.detach().numpy(), np.clip(self.accum_a.detach().numpy(), 0, self.accum_a.max()*0.6), rcount=self.accum_a.shape[0], vmin=0, cmap='viridis')
        if lim is not None:
            ax.set_zlim(bottom=0, top=lim)
        ax.set_xlabel('token')
        ax.set_ylabel('channel')
        ax.ticklabel_format(axis='z', style='sci', scilimits=(0,0))
        ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.zaxis.set_major_locator(MaxNLocator(5, integer=True))
        plt.title(png_name)
        plt.savefig(os.path.join(pngsave_dir,(png_name+'.png')))
        plt.cla()
        plt.clf()
        plt.close()


class Histogram():
    def __init__(self):
        super(Histogram, self).__init__()
        self.accum_a = torch.Tensor(0).to('cuda')
        self.accum_b = torch.Tensor(0).to('cuda')

        plt.rc('font', size=11)        
        plt.rc('axes', labelsize=11)   
        plt.rc('xtick', labelsize=11)  
        plt.rc('ytick', labelsize=11)  
        plt.rc('legend', fontsize=11)  
        plt.rc('figure', titlesize=11)
    
    def accumulate_a(self, original):
        flat_original = torch.flatten(original)
        self.accum_a = torch.concat((self.accum_a, flat_original), 0)

    def accumulate_b(self, quantized):
        flat_quantized = torch.flatten(quantized)
        self.accum_b = torch.concat((self.accum_b, flat_quantized), 0)

    def clear_a(self):
        self.accum_a = torch.Tensor(0).to('cuda')
    
    def clear_b(self):
        self.accum_b = torch.Tensor(0).to('cuda')

    def get_a(self):
        return self.accum_a
    
    def get_b(self):
        return self.accum_b
    
    def draw_plot(self, run_name, png_name, qerror, range=None, axvline=None, y_lim=None):
        pngsave_dir = os.path.join('./png',run_name)
        if not os.path.exists(pngsave_dir):
            os.mkdir(pngsave_dir)
        plt.hist(self.accum_a.detach().to('cpu').numpy(), bins=100, range=range, density=True, align='left', rwidth=0.8, color='c', label='original')
        plt.hist(self.accum_b.detach().to('cpu').numpy(), bins=100, range=range, density=True, align='left', rwidth=0.8, color='m', label='quantized')
        plt.legend(title=f"Quant error: {'%.2E' % Decimal(qerror)}")
        plt.yscale('log')
        plt.xlabel('Values')
        plt.ylabel('Density')
        if y_lim is not None:
            plt.ylim(y_lim)
        if axvline is not None:
            plt.axvline(x=axvline, color='b')

        plt.title(png_name)
        plt.savefig(os.path.join(pngsave_dir,(png_name+'.png')))

        bottom, top = plt.ylim()
        plt.clf()
        return bottom, top

class Tensor_Dict():
    def __init__(self):
        super(Tensor_Dict, self).__init__()
        self.module_dict = defaultdict(partial(defaultdict, torch.Tensor))
        self.idx = 0
    
    def key_init(self, module_name):
        self.module_dict[module_name]['weight'] = torch.zeros(1,1).to('cuda')
        self.module_dict[module_name]['input'] = torch.zeros(1,1).to('cuda')
        self.module_dict[module_name]['grad_out'] = torch.zeros(1,1).to('cuda')
    
    def register(self, tensor_name, input_tensor, direction='forward'):
        if not torch.equal(self.module_dict[list(self.module_dict)[self.idx]][tensor_name], torch.zeros(1,1).to('cuda')):
            return
        
        if self.idx > (len(list(self.module_dict))-1):
            raise Exception('idx is too big: ' + str(self.idx))
        elif self.idx < 0:
            raise Exception('idx is too small: ' + str(self.idx))
        
        self.module_dict[list(self.module_dict)[self.idx]][tensor_name] = input_tensor

        if direction == 'forward' and tensor_name == 'weight':
            None
        elif direction == 'forward' and tensor_name == 'input':
            if self.idx < len(list(self.module_dict))-1:
                self.idx += 1
        elif direction == 'backward' and tensor_name == 'grad_out':
            if self.idx > 0:
                self.idx -= 1
        else:
            raise NotImplementedError

    def get(self, module_name, tensor_name):
        return self.module_dict[module_name][tensor_name]
    
    def value_clear(self):
        for key, _ in self.module_dict.items():
            self.module_dict[key]['weight'] = torch.zeros(1,1).to('cuda')
            self.module_dict[key]['input'] = torch.zeros(1,1).to('cuda')
            self.module_dict[key]['grad_out'] = torch.zeros(1,1).to('cuda')

    def pickle_save(self, epoch, wagsave_dir, run_name):
        if not os.path.exists(wagsave_dir):
            os.mkdir(wagsave_dir)

        if not os.path.exists(os.path.join(wagsave_dir,run_name)):
            os.mkdir(os.path.join(wagsave_dir,run_name))
        
        filename = run_name + str(epoch) + '.pickle'
        filepath = os.path.join(os.path.join(wagsave_dir,run_name), filename)

        with open(file = filepath, mode='wb') as f:
            pickle.dump(self.module_dict, f)


class Tensor_Dict2():
    def __init__(self):
        super(Tensor_Dict2, self).__init__()
        self.tensor_dict = defaultdict(partial(defaultdict, torch.Tensor))
    
    def register(self, layer_name: str, tensor_name: str, input_tensor: torch.Tensor):
        if tensor_name != 'gx' or tensor_name != 'gw':
            return

        self.tensor_dict[layer_name][tensor_name] = input_tensor
        print(f'{layer_name}-{tensor_name} saved')

    def get(self, layer_name, tensor_name):
        return self.tensor_dict[layer_name][tensor_name]
    
    def value_clear(self):
        for key, _ in self.tensor_dict.items():
            self.tensor_dict[key]['gx'] = torch.zeros(1,1).to('cuda')
            self.tensor_dict[key]['gw'] = torch.zeros(1,1).to('cuda')

    def pickle_save(self, wagsave_dir, model_name, run_name):
        if not os.path.exists(wagsave_dir):
            os.mkdir(wagsave_dir)

        if not os.path.exists(os.path.join(wagsave_dir, model_name+"_gx_gw")):
            os.mkdir(os.path.join(wagsave_dir, model_name+"_gx_gw"))

        filename = run_name + '.pickle'
        filepath = os.path.join(os.path.join(wagsave_dir, model_name+"_gx_gw"), filename)

        with open(file = filepath, mode='wb') as f:
            pickle.dump(self.tensor_dict, f)
