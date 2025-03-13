import math
import numpy as np
import scipy
import torch
import random
from collections import defaultdict

# find biggest power2 factor under tensor size
def biggest_power2_factor(size_m):
    factors = []
    for i in range(1, size_m + 1):
        if size_m % i == 0:
            if math.log(i, 2).is_integer():
                factors.append(i)
    return max(factors)

def biggest_power2_factor_max16(size_m):
    factors = []
    for i in range(1, size_m + 1):
        if size_m % i == 0:
            if math.log(i, 2).is_integer():
                factors.append(i)

    if max(factors) > 16 and size_m > 16:
        return 16 # block size of diagonal elements
    else:
        return max(factors)

'''
1 1 1 -1 hadamard
'''
def hadamard(n):
    if n < 1:
        lg2 = 0
    else:
        lg2 = int(math.log(n, 2))
    if 2 ** lg2 != n:
        raise ValueError("n must be int and power of 2.")

    if n == 1:
        return np.array([[1]], dtype=int)

    H = np.array([[1,1],[1,-1]], dtype=int)

    # Hadamard stacking via Sylvester's construction
    # H H
    # H -H
    for i in range(0, lg2-1):
        H = np.vstack((np.hstack((H,H)),np.hstack((H,-H))))

    return H

def make_hadamard_block_diagonal(size_m):
    biggest_power2 = biggest_power2_factor(size_m)
    return scipy.linalg.block_diag(*[hadamard(biggest_power2)] * int(size_m / biggest_power2))

def make_hadamard(size_m):
    biggest_power2 = biggest_power2_factor_max16(size_m)
    return hadamard(biggest_power2)


'''
0 -1 -1 0 hadamard
'''

def adv_hadamard(n):
    if n < 1:
        lg2 = 0
    else:
        lg2 = int(math.log(n, 2))
    if 2 ** lg2 != n:
        raise ValueError("n must be int and power of 2.")

    if n == 1:
        return np.array([[1]], dtype=int)
    
    H = np.array([[0,-1],[-1,0]], dtype=int)

    # Hadamard stacking via Sylvester's construction
    # H H
    # H -H
    for i in range(0, lg2-1):
        H = np.vstack((np.hstack((H,H)),np.hstack((H,-H))))

    return H

def make_adv_hadamard(size_m):
    biggest_power2 = biggest_power2_factor_max16(size_m)
    return adv_hadamard(biggest_power2)

'''
haar
'''

def haar(n, normalized=False):
    # Allow only size n of power 2
    n = 2**np.ceil(np.log2(n))
    if n > 2:
        h = haar(n / 2)
    else:
        return np.array([[1, 1], [1, -1]])

    # calculate upper haar part
    h_n = np.kron(h, [1, 1])
    # calculate lower haar part 
    if normalized:
        h_i = np.sqrt(n/2)*np.kron(np.eye(len(h)), [1, -1])
    else:
        h_i = np.kron(np.eye(len(h)), [1, -1])
    # combine parts
    h = np.vstack((h_n, h_i))
    return h

def make_haar(size_m):
    biggest_power2 = biggest_power2_factor_max16(size_m)
    return scipy.linalg.block_diag(*[haar(biggest_power2)] * int(size_m / biggest_power2))

'''
Kronecker 4
'''
def lowering_matmul_front(A, H):
    H_h = H.shape[0]
    H_w = H.shape[1]
    if len(A.shape) == 2:
        A_h = A.shape[0]
        A_w = A.shape[1]
        if A_w == H_h:
            A = A @ H
        else:
            A = (A.reshape(-1, H.shape[0]) @ H).reshape(A_h, -1)
    elif len(A.shape) == 3:
        Batch_size = A.shape[0]
        A_h = A.shape[1]
        A_w = A.shape[2]
        if A_w == H_h:
            A = A @ H.unsqueeze(dim=0)
        else:
            A = (A.reshape(Batch_size, -1, H.shape[0]) @ H).reshape(Batch_size, A_h, -1)
    else:
        raise NotImplementedError
    return A

def lowering_matmul_back(H, A):
    H_h = H.shape[0]
    H_w = H.shape[1]
    if len(A.shape) == 2:
        A_h = A.shape[0]
        A_w = A.shape[1]
        if H_w == A_h:
            A = H @ A
        else:
            A = (A.T.reshape(-1, H.shape[1]) @ H.T).reshape(A_w, -1).T
    elif len(A.shape) == 3:
        Batch_size = A.shape[0]
        A_h = A.shape[1]
        A_w = A.shape[2]
        if H_w == A_h:
            A = H.unsqueeze(dim=0) @ A
        else:
            A = (A.mT.reshape(Batch_size, -1, H.shape[1]) @ H.mT).reshape(Batch_size, A_w, -1).mT
    return A

""" 
Walsh Hadamard transform
"""

def make_non_block_hadamard(size_m):
        if size_m == 1:
            return np.array([[1]])
        return hadamard(biggest_power2_factor(size_m))

def biggest_sqrt_factor(size_m):
        factors = []
        for i in range(1, size_m + 1):
            if np.sqrt(i).is_integer():
                if math.log(i, 2).is_integer():
                    factors.append(i)
        return max(factors)

def make_reorder_idx(size_m, percentile):
    def reorder(H):
        def sequency(row):
            return np.sum(row[:-1] != row[1:])
        reordered_idx = []
        for i in range(H.shape[0]):
            reordered_idx.append(sequency(H[i]))
        return reordered_idx
    
    reordered_idx = reorder(make_non_block_hadamard(size_m))
    sqrt = int(np.sqrt(biggest_sqrt_factor(size_m)))
    scale_factor = int(0.1*(-np.sqrt(2*sqrt**2*percentile+25)+10*sqrt+5))

    selected_idx = []
    if len(reordered_idx) == 1:
        return selected_idx

    for i in range(sqrt):
        if i % 2 == 0:
            row_idx_start = sqrt*i
            row_idx_end = sqrt*(i+1)-(i+scale_factor)
        elif i % 2 == 1:
            row_idx_start = sqrt*i+(i+scale_factor)
            row_idx_end = sqrt*(i+1)
        for j in range(row_idx_start, row_idx_end):
            selected_idx.append(reordered_idx.index(j))
    
    return selected_idx

def make_WHT(size_m, percentile):
    def reorder(H):
        def sequency(row):
            return np.sum(row[:-1] != row[1:])
        return np.array(sorted(H, key=sequency))
    
    wht_basis = reorder(make_non_block_hadamard(size_m))
    sqrt = int(np.sqrt(biggest_sqrt_factor(wht_basis.shape[0])))
    scale_factor = int(0.1*(-np.sqrt(2*sqrt**2*percentile+25)+10*sqrt+5))
    
    for i in range(sqrt):
        if i == 0:
            result_wht = wht_basis[sqrt*i:sqrt*(i+1)-(i+scale_factor), :]
        elif i % 2 == 0:
            result_wht = np.vstack((result_wht, wht_basis[sqrt*i:sqrt*(i+1)-(i+scale_factor), :]))
        elif i % 2 == 1:
            result_wht = np.vstack((result_wht, wht_basis[sqrt*i+(i+scale_factor):sqrt*(i+1), :]))
    return result_wht

class Transform_Dict():
    def __init__(self):
        self.transform_dict = defaultdict(np.array)

    def get(self, name, size):
        # print("get")
        return self.transform_dict[name+str(size)]

    def register(self, name, size, vectorPercentile):
        # print("register")
        if name == 'hadamard':
            self.transform_dict[name+str(size)] = make_hadamard(size)
        elif name == 'adv_hadamard':
            self.transform_dict[name+str(size)] = make_adv_hadamard(size)
        elif name == 'low_rank':
            self.transform_dict[name+str(size)] = make_WHT(size, vectorPercentile)
        elif name == 'reorder_idx':
            self.transform_dict[name+str(size)] = make_reorder_idx(size, vectorPercentile)
        else:
            raise NotImplementedError
        
    def get_or_register(self, name, size, vectorPercentile=50):
        if name+str(size) in self.transform_dict:
            return self.get(name, size)
        else:
            self.register(name, size, vectorPercentile)
            return self.get(name, size)