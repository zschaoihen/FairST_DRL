import os
import logging

import gin
from tqdm import tqdm

import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

@gin.configurable('BaseSolver')
class BaseSolver(object):
    # The super class for all solver class, describe all essential functions.
    def __init__(self, 
                use_cuda=gin.REQUIRED, max_iter=gin.REQUIRED,
                print_iter=gin.REQUIRED, batch_size=gin.REQUIRED):
        super(BaseSolver, self).__init__()

        # Essential parameters readin
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.max_iter = max_iter
        self.print_iter = print_iter
        self.global_iter = 0
        self.batch_size = batch_size
        self.pbar = tqdm(total=self.max_iter)

    def train(self):
        pass
    
    def viz_init(self):
        pass

    def visualize_recon(self):
        pass

    def visualize_line(self):
        pass

    def visualize_traverse(self):
        pass

    def net_mode(self, train_flag, nets):
        if not isinstance(train_flag, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in nets:
            if train_flag:
                net.train()
            else:
                net.eval()
    
    def save_checkpoint(self, ckptname='last', verbose=True):
        pass

    def load_checkpoint(self, ckptname='last', verbose=True):
        pass


class ModelFilter(logging.Filter):
    def __init__(self, model_name):
        super(ModelFilter, self).__init__()
        self.model_name = model_name

    def filter(self, record):
        return record.msg.startswith(self.model_name)