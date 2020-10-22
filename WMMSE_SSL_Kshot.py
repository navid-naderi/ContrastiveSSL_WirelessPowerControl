#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from itertools import chain, combinations
import pylab as pl
import time
from utils import *
from tqdm import tqdm
import random

def k_shot_SSL_train_test(
                    n = 10, # number of TP-UE pairs
                    num_train_samples = 5000, # total number of train samples
                    num_test_samples = 1000, # number of test samples
                    num_labeled_train_samples = 50, # number of labeled train samples
                    Pmax = 1, # max tx power
                    Pmin = 0, # min tx power
                    var_noise = 1, # Gaussian noise variance
                    batch_size = 256, # train/test batch size
                    hidden_layers = [256] * 3, # backbone hidden layers
                    num_SSL_pretrain_epochs = 10, # number of SSL pre-training epochs; if set to zero, SSL loss is never used
                    num_k_shot_epochs = 20, # number of k-shot SL training epochs
                    tau = 0.5, # temperatue in contrastive loss
                    lr = 0.001, # learning rate
                    device = 'cpu', # the device (cpu/gpu) to perform computations on
                    seed = 1234567, # random seed
                    ):

    # prepare the data
    num_data_points = num_train_samples + num_test_samples
    X_all, Y_all, total_time = generate_Gaussian(K=n, num_H=num_data_points, Pmax=Pmax, Pmin=Pmin, var_noise=var_noise)
    
    # set the random seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    phases = ['train-labeled', 'train-unlabeled', 'test']
    batch_sizes = {phase: batch_size for phase in phases}
    shuffles = {'train-labeled': True, 'train-unlabeled': True, 'test': False}

    X = dict()
    Y = dict()

    X['train-labeled'], Y['train-labeled'] = X_all.T[:num_labeled_train_samples], Y_all.T[:num_labeled_train_samples]

    X['train-unlabeled'], Y['train-unlabeled'] = X_all.T[num_labeled_train_samples:-num_test_samples], Y_all.T[num_labeled_train_samples:-num_test_samples]

    X['test'], Y['test'] = X_all.T[-num_test_samples:], Y_all.T[-num_test_samples:]
    
    
    data_sets = {phase: torch.utils.data.TensorDataset(torch.Tensor(X[phase]),
                                                       torch.Tensor(Y[phase])) for phase in phases}

    data_loaders = {phase: torch.utils.data.DataLoader(data_sets[phase],
                                                       batch_size=batch_sizes[phase],
                                                       shuffle=shuffles[phase]) for phase in phases}

    # prepare the backbone and PC head
    num_inputs = n ** 2
    non_linearity = nn.LeakyReLU()
    backbone_layers = nn.ModuleList()
    backbone_layers.append(nn.Linear(num_inputs, hidden_layers[0]))
    backbone_layers.append(non_linearity)
    for layer in range(1, len(hidden_layers)):
        backbone_layers.append(nn.Linear(hidden_layers[layer-1], hidden_layers[layer]))
        
        if layer < len(hidden_layers) - 1:
            backbone_layers.append(non_linearity)
        
    backbone = nn.Sequential(*backbone_layers).to(device)
    backbone.train()

    PC_layer = nn.Sequential(nn.Linear(hidden_layers[-1], n),
                             nn.Sigmoid()).to(device)
    PC_layer.train()

    criterion_MSE = torch.nn.MSELoss()
    criterion_CE = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(PC_layer.parameters()), lr = lr)


    epoch_losses = defaultdict(list)
    epoch_sum_rates = defaultdict(list)

    num_epochs = num_SSL_pretrain_epochs + num_k_shot_epochs

    for epoch in range(num_epochs):

        for phase in data_loaders:
            
            # ignore the unlabeled data in case no SSL is needed
            if num_SSL_pretrain_epochs == 0 and phase == 'train-unlabeled':
                continue
            
            if 'train' in phase:
                backbone.train()
                PC_layer.train()
            elif phase == 'test':
                backbone.eval()
                PC_layer.eval()
                X_test = torch.Tensor(0).to(device)
                Y_test = torch.Tensor(0).to(device)
                Y_test_pred = torch.Tensor(0).to(device)
            else:
                raise Exception

            losses = defaultdict(list)
            for i, data in enumerate(data_loaders[phase], 0):

                # get the inputs; data is a list of [inputs, labels]
                h, p = data
                
                h = h.to(device)
                p = p.to(device)

                optimizer.zero_grad()
                
                # contrastive SSL
                h_aug_1 = augment_ITLQ(h).to(device)
                h_aug_2 = augment_ITLQ(h).to(device)

                embeddings_1 = F.normalize(backbone(h_aug_1), dim=1)
                embeddings_2 = F.normalize(backbone(h_aug_2), dim=1)

                logits = torch.matmul(embeddings_1, embeddings_2.T) / tau

                labels = torch.Tensor(list(range(len(logits)))).to(device=device, dtype=torch.long)

                loss_NCE = criterion_CE(logits, labels)

                # Forward pass
                if num_SSL_pretrain_epochs > 0:
                    embeddings = F.normalize(backbone(h), dim=1)
                else:
                    embeddings = non_linearity(backbone(h))
                power_levels = PC_layer(embeddings)

                if phase == 'test':
                    X_test = torch.cat((X_test, h), dim=0)
                    Y_test = torch.cat((Y_test, p), dim=0)
                    Y_test_pred = torch.cat((Y_test_pred, power_levels), dim=0)

                loss_MSE = criterion_MSE(power_levels, p)

                # Compute the total loss
                if num_SSL_pretrain_epochs > 0: # SSL loss included
                    if epoch >= num_SSL_pretrain_epochs: # pre-training done, k-shot loss included
                        loss = loss_NCE + loss_MSE
                    else: # only pre-training SSL loss
                        loss = loss_NCE
                else: # SSL loss excluded
                    if phase == 'train-unlabeled': # no labels, hence no loss
                        loss = torch.Tensor([0]).to(device)
                    else: # only k-shot MSE loss included
                        loss = loss_MSE
                    
                
                losses['total'].append(loss.item())
                losses['NCE'].append(loss_NCE.item())
                losses['MSE'].append(loss_MSE.item())

                if 'train' in phase and loss > 0:
                    # Backward pass
                    loss.backward()
                    optimizer.step()


            if phase == 'test':
                test_data = {'X': X_test.detach().cpu().numpy(),
                             'Y': Y_test.detach().cpu().numpy(),
                             'Y_pred': Y_test_pred.detach().cpu().numpy(),
                            }
                return_baselines = (epoch == 0) # only calculate the sum-rates of baseline algorithms in the first epoch
                sum_rates = process_results(test_data, Pmax, var_noise, return_baselines=return_baselines)
                for alg in sum_rates:
                    epoch_sum_rates[alg].append(sum_rates[alg])

            for key in losses:
                epoch_losses[phase, key].append(np.mean(losses[key]))
                
            
            
    # repeat the baseline sum-rates over all epochs
    for alg in epoch_sum_rates:
        if len(epoch_sum_rates[alg]) == 1:
            epoch_sum_rates[alg] = epoch_sum_rates[alg] * len(epoch_sum_rates[alg])
    
    
    return epoch_losses, epoch_sum_rates
