import random
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import tempfile
import os
from torch import optim
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import yaml
import sys
sys.path.append("../utils")
from utils_trainer import trainer_wBert
import utils_dataset
import utils_builder

import wandb


os.environ["TOKENIZERS_PARALLELISM"] = "true"


def ddp_main():
    dist.init_process_group("nccl")
    torch.cuda.empty_cache()
    rank = dist.get_rank()

    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()

    # set up
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    if device_id == 0:
        run = wandb.init(
            # Set the project where this run will be logged
            project="MERL_ICML",
            name = config['wandb_name'],
            # Track hyperparameters and run metadata
            config={
                    "learning_rate": config['optimizer']['params']['lr'],
                    "total_epochs": config['trainer']['max_epochs'],
                    'weight_decay': config['optimizer']['params']['weight_decay'],
                    'ecg_model': config['network']['ecg_model'],
                    'text_model': config['network']['text_model'],
                    'batch_size': config['trainer']['batch_size'],
                    'val_zeroshot': 'all_sets',
                    'prompt_type': config['zeroshot']['prompt_type'],
            }
        )

    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)
    # loading data path
    data_path = config['dataset']['data_path']

    # define image-text dataset
    dataset = utils_dataset.ECG_TEXT_Dsataset(
        data_path=data_path, dataset_name=config['dataset']['dataset_name'])
    train_dataset = dataset.get_dataset(train_test='train')
    val_dataset = dataset.get_dataset(train_test='val')

    # building model part
    # --------------------
    model = utils_builder.ECGCLIP(config['network'])
    
    '''
    you can freeze bert from last layer to first layer.
    set num of layer in config.yaml
    default is freeze 9 layers
    '''
    if config['network']['free_layers'] is not None:
        for layer_idx in range(int(config['network']['free_layers'])):
            for param in list(model.lm_model.encoder.layer[layer_idx].parameters()):
                param.requires_grad = False

    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    # --------------------

    # --------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **config['optimizer']['params'],
        betas=(0.9, 0.999)
    )

    # ---------xw-----------
    trainer = trainer_wBert(model=model,
                            optimizer=optimizer,
                            device=rank,
                            model_name=config['wandb_name'],
                            **config['trainer'])
    # --------------------
    
    # --------------------
    # I_T_P_trainer
    trainer.train_w_TextEmb(train_dataset, val_dataset, config['zeroshot'])


ddp_main()
