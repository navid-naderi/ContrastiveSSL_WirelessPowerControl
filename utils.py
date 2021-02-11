import numpy as np
import math
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import defaultdict
from itertools import chain, combinations
import pylab as pl
from IPython import display
import time

# Functions for WMMSE algorithm
# Adapted from https://github.com/Haoran-S/SPAWC2017
def WMMSE_sum_rate(p_int, H, Pmax, var_noise):
    K = np.size(p_int)
    vnew = 0
    b = np.sqrt(p_int)
    f = np.zeros(K)
    w = np.zeros(K)
    for i in range(K):
        f[i] = H[i, i] * b[i] / (np.square(H[i, :]) @ np.square(b) + var_noise)
        w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
        vnew = vnew + math.log2(w[i])

    VV = np.zeros(100)
    for iter in range(100):
        vold = vnew
        for i in range(K):
            btmp = w[i] * f[i] * H[i, i] / sum(w * np.square(f) * np.square(H[:, i]))
            b[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp

        vnew = 0
        for i in range(K):
            f[i] = H[i, i] * b[i] / ((np.square(H[i, :])) @ (np.square(b)) + var_noise)
            w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
            vnew = vnew + math.log2(w[i])

        VV[iter] = vnew
        if vnew - vold <= 1e-5:
            break
            

    p_opt = np.square(b) / Pmax
    return p_opt

# Functions for data generation, Gaussian IC case
# Adapted from https://github.com/Haoran-S/SPAWC2017
def generate_Gaussian(K, num_H, Pmax=1, Pmin=0, var_noise=1, seed=420):
    np.random.seed(seed)
    Pini = Pmax*np.ones(K)
    X=np.zeros((K**2,num_H))
    Y=np.zeros((K,num_H))
    total_time = 0.0
    for loop in range(num_H):
        CH = 1/np.sqrt(2)*(np.random.randn(K,K)+1j*np.random.randn(K,K))
        H=abs(CH)
        X[:,loop] = np.reshape(H, (K**2,), order="F")
        H=np.reshape(X[:,loop], (K,K), order="F")
        mid_time = time.time()
        Y[:,loop] = WMMSE_sum_rate(Pini, H, Pmax, var_noise)
        total_time = total_time + time.time() - mid_time
        
        
    return X, Y, total_time

def sum_rate(H, p, Pmax, var_noise):
    n = H.shape[0]
    rate = 0
    for i in range(n):
        S = p[i] * Pmax * (H[i, i] ** 2)
        I = var_noise + np.sum(p * Pmax * (H[i, :] ** 2)) - S
        rate += np.log2(1 + S / I)
    return rate

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def augment_ITLQ(h, gamma=1, eta=0.5):
    h_t = np.copy(h.detach().cpu().numpy().T)
    N, num_samples = np.shape(h_t)
    n = int(np.sqrt(N))

    output = np.zeros_like(h_t)
    for sample in range(num_samples):
        H = h_t[:, sample].reshape(n, n, order="F")
        for i in range(n):
            for j in np.setdiff1d(range(n), i):
                if gamma * (H[i, j] ** 2) < (gamma * min(H[i, i], H[j, j]) ** 2) ** eta:
                    H[i, j] *= np.random.binomial(n=1, p=0.5)
        output[:, sample] = H.reshape(-1, order="F")
    return torch.Tensor(output.T)

def process_results(test_data, Pmax, var_noise, return_baselines=False):
    X, Y = test_data['X'].T, test_data['Y'].T
    if 'Y_pred' in test_data:
        Y_pred = test_data['Y_pred'].T
    else:
        Y_pred = None

    sum_rates = defaultdict(list)
    for sample in range(X.shape[1]):
        n = int(np.sqrt(X.shape[0]))
        H = X[:, sample].reshape(n, n, order="F")
        if Y_pred is not None:
            sum_rates['SSL'].append(sum_rate(H, Y_pred[:, sample], Pmax, var_noise))
        if return_baselines:
            sum_rates['WMMSE'].append(sum_rate(H, Y[:, sample], Pmax, var_noise))
            sum_rates['FR'].append(sum_rate(H, np.ones(n), Pmax, var_noise))

    for alg in sum_rates:
        sum_rates[alg] = np.mean(sum_rates[alg])
        
    return sum_rates