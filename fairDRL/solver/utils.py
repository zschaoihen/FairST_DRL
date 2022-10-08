import os
import math
import logging

import gin
from tqdm import tqdm

import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import distributions as dist

class DataGather(object):
    def __init__(self, *args):
        self.keys = args
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return {arg:[] for arg in self.keys}

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()

def recon_loss(x, x_recon):
    '''
        Calculate the binary cross entropy between recon and x.
        Noted that it use 'binary_cross_entropy_with_logits' which 
        means the decoder doesn't need  a sigmoid layer.
    '''
    batch_size = x.size(0)
    assert batch_size != 0
    flat_recon = torch.flatten(x_recon, start_dim=1)
    flat_input = torch.flatten(x, start_dim=1)

    # x_recon = F.sigmoid(x_recon)
    loss = F.mse_loss(flat_input, flat_recon, size_average=False).div(batch_size)
    # loss = F.mse_loss(x, x_recon, size_average=False).div(batch_size)
    # loss = F.mse_loss(x_recon, x)
    # loss = F.l1_loss(x_recon, x, size_average=False).div(batch_size)
    return loss


def kl_divergence(mu, logvar):
    # Calculate the KL divergnece based on the mu and logvar.
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld


def permute_dims(z):
    # Dedicated for FactorVAE.
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))
