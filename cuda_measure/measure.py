import os
import sys
import gc

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import numpy as np
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

B = 49*256
O = 512
I = 4608

warmupTurn = 30
testTurn = 100
eps = 1e-11


def time_gpu(fn, n_iter, n_warmup):
    """Measure GPU latency of `fn` in milliseconds using CUDA events.

    Returns a list of per-iteration elapsed times (ms). Warmup runs are
    executed but not recorded so that context init, allocator warmup and
    clock ramp-up do not pollute the measurement.
    """
    # Warmup
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in tqdm(range(n_iter)):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))  # ms
    return times


def main():
    # Fixed inputs: generate once so H2D transfer / RNG do not enter the loop.
    X = torch.Tensor(np.random.lognormal(1., 0.1, B * I).reshape(B, I)).to('cuda')
    Go = torch.Tensor(np.random.lognormal(1., 0.1, B * O).reshape(B, O)).to('cuda')
    W = torch.Tensor(np.random.lognormal(1., 0.1, O * I).reshape(O, I)).to('cuda')

    # Full FP: Gi = Go @ W, Gw = Go.T @ X
    def run_fp():
        Gi_FP = torch.matmul(Go, W)
        Gw_FP = torch.matmul(Go.T, X)
        return Gi_FP, Gw_FP

    # Per-token quant: FWHT + Stoch. Q + INT4 Matmul
    def run_per_token():
        return HLQ_backward.backward(Go, W, X, 0, 0, 16, True)

    # Per-tensor quant: FWHT + Stoch. Q + INT4 Matmul
    def run_per_tensor():
        return HLQ_backward.backward(Go, W, X, 0, 0, 16, False)

    print("Measuring FP ...")
    FP_time = time_gpu(run_fp, testTurn, warmupTurn)
    print("Measuring per-token ...")
    per_token_time = time_gpu(run_per_token, testTurn, warmupTurn)
    print("Measuring per-tensor ...")
    per_tensor_time = time_gpu(run_per_tensor, testTurn, warmupTurn)

    print("")
    print(f"{statistics.median(FP_time):.8f} ms | FP")
    print(f"{statistics.median(per_tensor_time):.8f} ms | per_tensor")
    print(f"{statistics.median(per_token_time):.8f} ms | per_token")
    print("")


if __name__ == '__main__':

    main()
