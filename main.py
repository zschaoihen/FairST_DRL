import os
import gc
import json 
import argparse
import logging
import logging.handlers
import copy
import gin

import numpy as np
import torch

from fairDRL.utils import str2bool, mkdirs
from fairDRL.exps import *
from fairDRL.solvers import *

@gin.configurable('single_run', blacklist=['seed', 'exp_models', 'logger', 
                                            'gamma', 'log_dir', 'ckpt_dir'])
def single_run(seed, exp_models, logger, gamma, log_dir, ckpt_dir, eval_metrics=gin.REQUIRED, mode=gin.REQUIRED):
    temp_log_dir = log_dir + '{}_{}/'.format(seed, gamma)
    temp_ckpt_dir = ckpt_dir + '{}_{}/'.format(seed, gamma)
    mkdirs(temp_log_dir)
    mkdirs(temp_ckpt_dir)

    for model in exp_models:
        # temp_module_name = "{}Solver".format(model)
        # temp_func_name = temp_module_name[0].upper() + temp_module_name[1:]
        # if temp_module_name in globals():
        #     model_solver = globals()[temp_module_name].__dict__[temp_func_name](mode, temp_log_dir, temp_ckpt_dir)
        #     model_solver = single_training(seed, model_solver, logger, gamma, temp_log_dir, temp_ckpt_dir)
        # else:
        #     print("no such solver: {} solver found in this package".format(model))

        solver_func = solver_getter(model)
        model_solver = solver_func(mode, temp_log_dir, temp_ckpt_dir)
        single_training(seed, model_solver, logger, gamma, temp_log_dir, temp_ckpt_dir)
        # model_solver.clean()

        del model_solver


@gin.configurable('main')
def main(base_path, run_number, exp_models, gammas, seeds):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    result_path = base_path + 'results/{}/'.format(run_number)
    ckpt_dir = result_path + 'models/'
    log_dir = result_path + 'log/'

    mkdirs(ckpt_dir)
    mkdirs(log_dir)

    logger = logging.getLogger('rootlogger')
    logger.setLevel(logging.DEBUG)
    
    root_handler = logging.FileHandler(log_dir+'root_log.log')
    root_handler.setLevel(logging.INFO)
    root_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(root_handler)
    logger.addHandler(logging.StreamHandler())

    # print('Exp starts.')
    logger.info('Exp starts.')

    for seed in seeds:
        for gamma in gammas:
            single_run(seed, exp_models, logger, gamma, log_dir, ckpt_dir) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fair-VAE')
    parser.add_argument('--config_name', default=None, type=str, help='the config file path')
    args = parser.parse_args()

    gin.parse_config_file('./configs/{}/main_setup.gin'.format(args.config_name))

    main()