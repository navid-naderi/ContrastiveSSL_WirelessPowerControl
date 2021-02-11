import numpy as np
import math
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import defaultdict
from itertools import chain, combinations, product
import time
from utils import *
from tqdm import tqdm
from WMMSE_SSL_Kshot import k_shot_SSL_train_test
import pickle
import argparse
from multiprocessing import Pool
import copy
import os

import seaborn as sns
sns.set_style("whitegrid")

def k_shot_SSL_train_test_nrange_krange(args):
    
    num_pairs_range = args.n # list of number of TP-UE pairs
    num_labeled_train_samples_range = args.k # list of number of labeled train samples
    num_train_samples = args.num_train_samples # total number of train samples
    num_test_samples = args.num_test_samples # number of test samples
    Pmax = args.Pmax # max tx power
    Pmin = args.Pmin # min tx power
    var_noise = args.var_noise # Gaussian noise variance
    batch_size = args.batch_size # train/test batch size
    hidden_layers = args.hidden_layers # backbone hidden layers
    
    # number of SSL pre-training epochs; if set to zero, SSL loss is never used
    num_SSL_pretrain_epochs = args.num_SSL_pretrain_epochs
    
    num_k_shot_epochs = args.num_k_shot_epochs # number of k-shot SL training epochs
    tau = args.tau # temperatue in contrastive loss
    lr = args.lr # learning rate
    device = args.device # the device (cpu/gpu) to perform computations on
    seed = args.seed # random seed
    
    # generate all (n, k) combinations
    n_k_combinations = list(product(num_pairs_range, num_labeled_train_samples_range))

    params = [(n, num_train_samples, num_test_samples, k, Pmax, Pmin, var_noise, batch_size, hidden_layers,\
                    num_SSL_pretrain_epochs, num_k_shot_epochs, tau, lr, device, seed) for (n, k) in n_k_combinations]

    pool = Pool()
    all_results = pool.starmap(k_shot_SSL_train_test, params)
    pool.close()
    pool.join()

    # process the results
    sum_rates = defaultdict(list)
    for n in num_pairs_range:
        for k in num_labeled_train_samples_range:
            result_index = n_k_combinations.index((n, k))
            epoch_losses, epoch_sum_rates = all_results[result_index][:2]
            for alg in epoch_sum_rates:
                sum_rates[n, alg].append(np.max(epoch_sum_rates[alg]))
                
    return sum_rates

def parse_option():
    
    parser = argparse.ArgumentParser('Contrastive self-supervised learning for wireless power control')

    parser.add_argument('--n', type=int, nargs='+', default=[6], help='List of number of TP-UE pairs')
    parser.add_argument('--k', type=int, nargs='+', default=[4, 8, 16, 32, 64, 128], help='List of number of labeled training samples')
    parser.add_argument('--num_train_samples', type=int, default=1000, help='Total number of training samples')
    parser.add_argument('--num_test_samples', type=int, default=1000, help='Total number of test samples')
    parser.add_argument('--Pmax', type=float, default=1, help='Maximum transmit power (in Watts)')
    parser.add_argument('--Pmin', type=float, default=0, help='Minimum transmit power (in Watts)')
    parser.add_argument('--var_noise', type=float, default=1, help='Additive Gaussian noise variance')
    parser.add_argument('--batch_size', type=int, default=64, help='Train/test batch size')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[512] * 2, help='List of backbone hidden layer sizes')
    parser.add_argument('--num_SSL_pretrain_epochs', type=int, default=20,
                        help='Number of SSL pre-training epochs; if set to zero, SSL loss is never used!')
    parser.add_argument('--num_k_shot_epochs', type=int, default=100, help='Number of few-shot supervised training epochs')
    parser.add_argument('--tau', type=float, default=0.1, help='Temperature parameter in contrastive loss')
    parser.add_argument('--lr', type=float, default=5e-2, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='The device (cpu/gpu) for performing computations')
    parser.add_argument('--seed', type=int, default=1234567, help='Random seed for reproducible results')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of experiments with different random seeds')
    
    opt = parser.parse_args()
    
    return opt

def main():
    
    args = parse_option()
    
    experiment_name = '{0}shot_results_{1}pairs_backbone{2}_tau{3}_batch{4}'.format(args.k,
                                                                                    args.n,
                                                                                    args.hidden_layers,
                                                                                    args.tau,
                                                                                    args.batch_size)
    
    all_sum_rates_SSL = defaultdict(list)
    all_sum_rates_noSSL = defaultdict(list)
    
    if args.num_SSL_pretrain_epochs > 0:
        print('Now running experiments with SSL pre-training ...')
    else:
        print('Now running experiments with only supervised training ...')
        
    initial_seed = copy.deepcopy(args.seed)
    for _ in tqdm(range(args.num_runs)):
        args.seed += 10 # change the random seed
        sum_rates = k_shot_SSL_train_test_nrange_krange(args)
        for key in sum_rates:
            all_sum_rates_SSL[key].append(sum_rates[key])
            
    if args.num_SSL_pretrain_epochs > 0: # also compare the performance with pure supervised training
        args.seed = initial_seed # revert the random seed to its original value
        print('Now running experiments with only supervised training ...')
        args.num_SSL_pretrain_epochs = 0 # remove the pre-training epochs from input args
        args.lr = 1e-2 # reduce the learning rate for pure supervised training
        for _ in tqdm(range(args.num_runs)):
            args.seed += 10 # change the random seed
            sum_rates = k_shot_SSL_train_test_nrange_krange(args)
            for key in sum_rates:
                all_sum_rates_noSSL[key].append(sum_rates[key])
                
    if not os.path.exists('results'):
        os.makedirs('results')
                
    with open('./results/{}.json'.format(experiment_name), 'wb') as handle:
        pickle.dump([args.n, args.k, all_sum_rates_SSL, all_sum_rates_noSSL], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('./results/{}.json'.format(experiment_name), 'rb') as handle:
        n_range, k_range, all_sum_rates_SSL, all_sum_rates_noSSL = pickle.load(handle)
        
    algs = ['SSL', 'SL only', 'WMMSE', 'FR']
    markers = {alg: '' for alg in algs}
    markers['SL only'] = 'o'
    markers['SSL'] = 's'
    
    for n in n_range:
        
        plot_name = '{0}shot_results_{1}pairs_backbone{2}_tau{3}_batch{4}'.format(k_range, n, args.hidden_layers,
                                                                                  args.tau, args.batch_size)
        
        x_range = 100 * (np.array(k_range) / args.num_train_samples)

        plt.figure()
        alpha = .2
        for alg in algs:
            if alg == 'SL only' and not ((n, 'SSL') in all_sum_rates_noSSL):
                continue
            elif alg == 'SL only' and (n, 'SSL') in all_sum_rates_noSSL:
                all_sum_rates_SSL[n, alg] = np.array(all_sum_rates_noSSL[n, 'SSL'])
            else:
                all_sum_rates_SSL[n, alg] = np.array(all_sum_rates_SSL[n, alg])
                
            plt.plot(x_range, np.mean(all_sum_rates_SSL[n, alg], axis=0), label=alg, marker=markers[alg])
            plt.fill_between(x_range, np.mean(all_sum_rates_SSL[n, alg], axis=0) - np.std(all_sum_rates_SSL[n, alg], axis=0),
                                        np.mean(all_sum_rates_SSL[n, alg], axis=0) + np.std(all_sum_rates_SSL[n, alg], axis=0),
                             alpha=alpha)

        plt.legend(loc='upper left', bbox_to_anchor=(0.0,0.9), fancybox=True)
        plt.grid(True)
        plt.xlabel('Fraction of labeled training samples (%)')
        plt.ylabel('Sum-rate (bps/Hz)')
        plt.gca().set_xscale('log')
        plt.savefig('./results/{}.pdf'.format(plot_name), bbox_inches='tight')
        plt.show()
    
if __name__ == '__main__':
    main()