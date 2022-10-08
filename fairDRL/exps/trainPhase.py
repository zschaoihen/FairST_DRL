import os
import json 
import argparse
import logging
import logging.handlers
import copy
import gin

import numpy as np
import torch

from ..utils import str2bool, mkdirs

def single_training(seed, model_solver, logger, gamma, log_dir, ckpt_dir):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    model_solver.gamma = gamma

    if model_solver.log_flag:
        model_solver.logger.debug('{}: set gamma to {}'.format(model_solver.model_name, gamma))
    if not model_solver.ckpt_load:
        model_solver.train()
        logger.info('{}: training stage finished.'.format(model_solver.model_name))
    if model_solver.log_flag:
        logger.removeHandler(model_solver.model_handler)

    # return model_solver