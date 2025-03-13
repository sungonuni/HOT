import os
import sys
import gc

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import numpy as np
import time
import math
import random
import statistics
from tqdm import tqdm

# import FLinearQ_backward
import HLQ_backward
# import fast_walsh_hadamard

from utils.hadamard_matmul import *

seed = 2023
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

B = 49
O = 3072
I = 768

testTurn = 100
eps = 1e-11

def main():

    FP_time = []
    per_tensor_time = []
    per_token_time = []
    

    for i in tqdm(range(testTurn)):
        X = torch.Tensor(np.random.lognormal(1., 0.1, B*I).reshape(B, I)).to('cuda')
        Go = torch.Tensor(np.random.lognormal(1., 0.1, B*O).reshape(B, O)).to('cuda')
        W = torch.Tensor(np.random.lognormal(1., 0.1, O*I).reshape(O, I)).to('cuda')

        # Full FP
        start = time.time()
        Gi_FP = torch.matmul(Go, W)
        torch.cuda.synchronize()
        Gw_FP = torch.matmul(Go.T, X) 
        torch.cuda.synchronize()
        FP_end = time.time()
        print(f"{FP_end - start} sec | FP")
        FP_time.append(time.time() - start)

        # Per-token quant
        Gi_H, Gw_tH, time_list1 = HLQ_backward.backward(Go, W, X, 0, 0, 16, True) # FWHT + Stoch. Q + INT4 Matmul
        torch.cuda.synchronize()
        per_token_time.append(sum(time_list1))
        print(f"{sum(time_list1)} sec | per_token")
        time_list1 = []

        # Per-tensor quant
        Gi_H, Gw_tH, time_list2 = HLQ_backward.backward(Go, W, X, 0, 0, 16, False) # FWHT + Stoch. Q + INT4 Matmul
        torch.cuda.synchronize()
        per_tensor_time.append(sum(time_list2))
        print(f"{sum(time_list2)} sec | per_tensor")
        time_list2 = []
    
    print(f"{statistics.median(FP_time):.8f} nsec | FP")
    print(f"{statistics.median(per_tensor_time):.8f} nsec | per_tensor")
    print(f"{statistics.median(per_token_time):.8f} nsec | per_token")
    

if __name__ == '__main__':

    main()