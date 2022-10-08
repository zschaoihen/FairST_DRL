# package fairDRL.solvers
# __init__.py

import os
import random
import math
import json
import logging
import argparse

import gin
import numpy as np
import pandas as pd 
import visdom
from tqdm import tqdm

import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .FairSolver import FairSolver

def solver_getter(model):
    if model == 'fairVAE':
        return fairSolver
    # elif model == 'factorVAE':
    #     return FactorVAESolver
    # elif model == 'betaVAE':
    #     return BetaVAESolver
    # elif model == 'VAE':
    #     return VAESolver
    else:
        raise NameError('Solver type not support!')