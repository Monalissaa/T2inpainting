#!/usr/bin/env python3

import imp
import logging
import os
import sys
import traceback

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import hydra
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from saicinpainting.training.trainers import load_checkpoint

from saicinpainting.training.trainers import make_training_model
from saicinpainting.utils import register_debug_signal_handlers, handle_ddp_subprocess, handle_ddp_parent_process, \
    handle_deterministic_config

LOGGER = logging.getLogger(__name__)
import torch
import numpy as np
import random
def seed_everything(seed):
    torch.manual_seed(seed) # Current CPU
    torch.cuda.manual_seed(seed) # Current GPU
    np.random.seed(seed) # Numpy module
    random.seed(seed) # Python random module
    torch.backends.cudnn.benchmark = False # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)

@handle_ddp_subprocess()
@hydra.main(config_path='../configs/training', config_name='tiny_test.yaml')
def main(config: OmegaConf):
    try:
        need_set_deterministic = handle_deterministic_config(config)

        register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        is_in_ddp_subprocess = handle_ddp_parent_process()

        config.visualizer.outdir = os.path.join(os.getcwd(), config.visualizer.outdir)
        if not is_in_ddp_subprocess:
            LOGGER.info(OmegaConf.to_yaml(config))
            OmegaConf.save(config, os.path.join(os.getcwd(), 'config.yaml'))

        checkpoints_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(checkpoints_dir, exist_ok=True)

        # there is no need to suppress this logger in ddp, because it handles rank on its own
        metrics_logger = TensorBoardLogger(config.location.tb_dir, name=os.path.basename(os.getcwd()))
        metrics_logger.log_hyperparams(config)

        if config.new_params.seed>0:
            seed_everything(config.new_params.seed)

        # training_model = make_training_model(config)
        # checkpoint_path = '/home/mona/codes/lama/experiments/celeb_hq_all_256_lama_from_scratch/mona_2022-01-11_11-09-25_train_lama-fourier-celeba_/models/best.ckpt'
        # checkpoint_path = '/home/mona/codes/lama/experiments/big-lama-transfer-reInitUp/10-36-13/models/best.ckpt'
        if config.new_params.two_stage_from_init:
            checkpoint_path = config.new_params.resume_from_checkpoint
        elif config.new_params.from_cwd:
            checkpoint_path = config.new_params.from_cwd_path
        elif config.new_params.celeba:
            checkpoint_path = os.environ.get('TORCH_HOME')+'/experiments/lama-fourier/models/best.ckpt'
        elif config.new_params.tsa.from_our1_path != False:
            checkpoint_path = config.new_params.tsa.from_our1_path
        else:
            checkpoint_path = os.environ.get('TORCH_HOME')+'/experiments/big-lama-with-discr/models/best.ckpt'

        if config.new_params.from_scratch:
            training_model = make_training_model(config)
        else:
            training_model = load_checkpoint(config, checkpoint_path)

        trainer_kwargs = OmegaConf.to_container(config.trainer.kwargs, resolve=True)
        if need_set_deterministic:
            trainer_kwargs['deterministic'] = True
        
        trainer = Trainer(
            # there is no need to suppress checkpointing in ddp, because it handles rank on its own
            callbacks=ModelCheckpoint(dirpath=checkpoints_dir, **config.trainer.checkpoint_kwargs),
            logger=metrics_logger,
            default_root_dir=os.getcwd(),
            **trainer_kwargs
        )

        trainer.fit(training_model)
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Training failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
