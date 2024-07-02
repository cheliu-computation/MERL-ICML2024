# package import
# import wandb
import os
from typing import Type

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as torch_dist
import torch.distributed as dist

from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import yaml as yaml

from utils_loss import clip_loss
from zeroshot_val import zeroshot_eval

import wandb

class trainer_wBert:
    def __init__(self, model,
                 optimizer, device, model_name, **args):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.train_batch_size = args['batch_size']
        self.max_epochs = args['max_epochs']
        self.num_workers = args['num_workers']
        self.checkpoint_interval = args['checkpoint_interval']
        self.val_batch_size = args['val_batch_size']

    # traing process
    def train_w_TextEmb(self, train_dataset, val_dataset, args_zeroshot_eval):

        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size,
                                  num_workers=self.num_workers,
                                  drop_last=True, shuffle=False,
                                  sampler=DistributedSampler(train_dataset))
        
        val_loader = DataLoader(val_dataset, batch_size=self.val_batch_size,
                                num_workers=self.num_workers,
                                drop_last=True, shuffle=False,
                                sampler=DistributedSampler(val_dataset))
                                
        
        model_checkpoints_folder = os.path.join('../checkpoints/')
        if not os.path.exists(model_checkpoints_folder):
            print('create directory "{}" for save checkpoint!'.format(
                model_checkpoints_folder))
            print('---------------------------')
            os.makedirs(model_checkpoints_folder)
        else:
            print('directory "{}" existing for save checkpoint!'.format(
                model_checkpoints_folder))

        # automatically resume from checkpoint if it exists
        print('#########################################')
        print('Be patient..., checking checkpoint now...')
        if os.path.exists(model_checkpoints_folder + self.model_name+'_checkpoint.pth'):
            ckpt = torch.load(model_checkpoints_folder + self.model_name+'_checkpoint.pth',
                              map_location='cpu')
            start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print('continue training successful!')
        else:
            start_epoch = 0
            print('Start training from 0 epoch')

        print('#########################################')
        print('training start!')

        # scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5000,
            T_mult=1,
            eta_min=1e-8,
        )
        niter = 1

        skip_scheduler = False
        scaler = GradScaler()

        f1_total = []
        acc_total = []
        auc_total = []

        zeroshot_csv = pd.DataFrame()
        best_auc = 0

        for epoch_counter in tqdm(range(start_epoch, self.max_epochs+1)):

            epoch_loss = 0
            epoch_acc1 = []
            epoch_acc5 = []
            self.model.train()
            for data in tqdm(train_loader):
                self.model.train()
                # get raw text
                report = data['raw_text']

                # get ecg
                ecg = data['ecg'].to(torch.float32).to(
                    self.device).contiguous()
                
                self.optimizer.zero_grad()

                with autocast():
                    report_tokenize_output = self.model.module._tokenize(report)

                    input_ids = report_tokenize_output.input_ids.to(
                        self.device).contiguous()
                    attention_mask = report_tokenize_output.attention_mask.to(
                        self.device).contiguous()

                    output_dict = self.model(ecg, input_ids, attention_mask) 
                    ecg_emb, proj_ecg_emb, proj_text_emb = output_dict['ecg_emb'],\
                                                            output_dict['proj_ecg_emb'],\
                                                            output_dict['proj_text_emb']


                    world_size = torch_dist.get_world_size()
                    with torch.no_grad():
                        agg_proj_img_emb = [torch.zeros_like(proj_ecg_emb[0]) for _ in range(world_size)]
                        agg_proj_text_emb = [torch.zeros_like(proj_text_emb[0]) for _ in range(world_size)]
                        
                        dist.all_gather(agg_proj_img_emb, proj_ecg_emb[0])
                        dist.all_gather(agg_proj_text_emb, proj_text_emb[0])
                        
                        agg_proj_ecg_emb1 = [torch.zeros_like(ecg_emb[0]) for _ in range(world_size)]
                        agg_proj_ecg_emb2 = [torch.zeros_like(ecg_emb[1]) for _ in range(world_size)]
                        dist.all_gather(agg_proj_ecg_emb1, ecg_emb[0])
                        dist.all_gather(agg_proj_ecg_emb2, ecg_emb[1])
                        # get current rank
                        rank = torch_dist.get_rank()

                    agg_proj_img_emb[rank] = proj_ecg_emb[0]
                    agg_proj_text_emb[rank] = proj_text_emb[0]
                    
                    agg_proj_ecg_emb1[rank] = ecg_emb[0]
                    agg_proj_ecg_emb2[rank] = ecg_emb[1]

                    agg_proj_img_emb = torch.cat(agg_proj_img_emb, dim=0)
                    agg_proj_text_emb = torch.cat(agg_proj_text_emb, dim=0)

                    agg_proj_ecg_emb1 = torch.cat(agg_proj_ecg_emb1, dim=0)
                    agg_proj_ecg_emb2 = torch.cat(agg_proj_ecg_emb2, dim=0)

                    cma_loss, acc1, acc5 = clip_loss(agg_proj_img_emb, agg_proj_text_emb, device=self.device)
                    uma_loss, _, _ = clip_loss(agg_proj_ecg_emb1, agg_proj_ecg_emb2, device=self.device)
                    loss = cma_loss + uma_loss

                    if self.device == 0:
                        print(f'loss is {loss.item()}, acc1 is {acc1.item()}, acc5 is {acc5.item()}, cma_loss is {cma_loss.item()}, uma_loss is {uma_loss.item()}')

                        wandb.log({
                            'train_step_uma_loss': uma_loss.item(),
                            'train_step_cma_loss': cma_loss.item(),
                            'train_step_total_loss': loss.item(),
                            'train_step_acc1': acc1.item(),
                            'train_step_acc5': acc5.item()}
                            )
                        
                    # accumalate loss for logging
                    epoch_loss += loss.item()
                    epoch_acc1.append(acc1.item())
                    epoch_acc5.append(acc5.item())

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    if not skip_scheduler:
                        scheduler.step()
                niter += 1

            # eval stage
            val_log = self.val(val_loader)
            
            if self.device == 0:
                # average train metric
                epoch_acc1 = np.array(epoch_acc1).mean()
                epoch_acc5 = np.array(epoch_acc5).mean()

                epoch_iter = (len(train_dataset)//self.train_batch_size)
                print(f'{epoch_counter} epoch loss is {epoch_loss/epoch_iter},\
                                    acc1 is {epoch_acc1}, acc5 is {epoch_acc5}')
                
                # log train and val epoch metric
                wandb.log({ 
                            'train_epoch_loss': epoch_loss/epoch_iter,
                            'train_epoch_acc1': epoch_acc1,
                            'train_epoch_acc5': epoch_acc5,
                            'val_cma_loss': val_log['val_cma_loss'],
                            'val_uma_loss': val_log['val_uma_loss'],
                            'val_epoch_loss': val_log['val_loss'],
                            'val_epoch_acc1': val_log['val_acc1'],
                            'val_epoch_acc5': val_log['val_acc5']}
                            )
                
                
                # zero-shot eval      
                avg_f1, avg_acc, avg_auc = 0, 0, 0
                for set_name in args_zeroshot_eval['val_sets'].keys():

                    f1, acc, auc, _, _, _, res_dict = \
                        zeroshot_eval(model=self.model, 
                        set_name=set_name, 
                        device=self.device, 
                        args_zeroshot_eval=args_zeroshot_eval)

                    avg_f1 += f1
                    avg_acc += acc
                    avg_auc += auc

                    # log each val set zeroshot performance
                    wandb.log({ 
                                f'{set_name}_f1': f1,
                                f'{set_name}_acc': acc,
                                f'{set_name}_AUROC': auc
                                }
                                )
                
                avg_f1 = avg_f1/len(args_zeroshot_eval['val_sets'].keys())
                avg_acc = avg_acc/len(args_zeroshot_eval['val_sets'].keys())
                avg_auc = avg_auc/len(args_zeroshot_eval['val_sets'].keys())
                wandb.log({
                            'avg_f1': avg_f1,
                            'avg_acc': avg_acc,
                            'avg_auc': avg_auc
                            }
                            )
                
                f1_total.append(f1)
                acc_total.append(acc)
                auc_total.append(auc)

                best_metric = avg_auc
                if best_metric > best_auc:
                    best_auc = best_metric
                    torch.save(self.model.module.state_dict(),
                               model_checkpoints_folder + self.model_name+f'_bestZeroShotAll_ckpt.pth')
                    torch.save(self.model.module.ecg_encoder.state_dict(),
                                model_checkpoints_folder + self.model_name+f'_bestZeroShotAll_encoder.pth')
                    
                if epoch_counter % self.checkpoint_interval == 0:
                    self.save_checkpoints(epoch_counter, model_checkpoints_folder + self.model_name + f'_{epoch_counter}_ckpt.pth')
            
        if self.checkpoint_interval != 1:
            # save final ecg_encoder
            torch.save(self.model.module.ecg_encoder.state_dict(),
                    model_checkpoints_folder + self.model_name + '_final_encoder.pth')
            # save final total model
            torch.save(self.model.module.state_dict(),
                    model_checkpoints_folder + self.model_name + '_final_total.pth')

    def val(self, loader):
        print('start validation')
        self.model.eval()
        val_cma_loss = 0
        val_uma_loss = 0
        val_loss = 0
        val_epoch_acc1 = []
        val_epoch_acc5 = []
        
        for data in tqdm(loader):
            # get raw text
            report = data['raw_text']
            # get ecg
            ecg = data['ecg'].to(torch.float32).to(
                self.device).contiguous()
            
            report_tokenize_output = self.model.module._tokenize(report)

            input_ids = report_tokenize_output.input_ids.to(
                self.device).contiguous()
            attention_mask = report_tokenize_output.attention_mask.to(
                self.device).contiguous()
            
            with torch.no_grad():
                output_dict = self.model(ecg, input_ids, attention_mask) 
                ecg_emb, proj_ecg_emb, proj_text_emb = output_dict['ecg_emb'],\
                                                            output_dict['proj_ecg_emb'],\
                                                            output_dict['proj_text_emb']


                world_size = torch_dist.get_world_size()
                with torch.no_grad():
                    agg_proj_img_emb = [torch.zeros_like(proj_ecg_emb[0]) for _ in range(world_size)]
                    agg_proj_text_emb = [torch.zeros_like(proj_text_emb[0]) for _ in range(world_size)]
                    
                    dist.all_gather(agg_proj_img_emb, proj_ecg_emb[0])
                    dist.all_gather(agg_proj_text_emb, proj_text_emb[0])
                    
                    agg_proj_ecg_emb1 = [torch.zeros_like(ecg_emb[0]) for _ in range(world_size)]
                    agg_proj_ecg_emb2 = [torch.zeros_like(ecg_emb[1]) for _ in range(world_size)]
                    dist.all_gather(agg_proj_ecg_emb1, ecg_emb[0])
                    dist.all_gather(agg_proj_ecg_emb2, ecg_emb[1])
                    # get current rank
                    rank = torch_dist.get_rank()

                agg_proj_img_emb[rank] = proj_ecg_emb[0]
                agg_proj_text_emb[rank] = proj_text_emb[0]
                
                agg_proj_ecg_emb1[rank] = ecg_emb[0]
                agg_proj_ecg_emb2[rank] = ecg_emb[1]

                agg_proj_img_emb = torch.cat(agg_proj_img_emb, dim=0)
                agg_proj_text_emb = torch.cat(agg_proj_text_emb, dim=0)

                agg_proj_ecg_emb1 = torch.cat(agg_proj_ecg_emb1, dim=0)
                agg_proj_ecg_emb2 = torch.cat(agg_proj_ecg_emb2, dim=0)

                cma_loss, acc1, acc5 = clip_loss(agg_proj_img_emb, agg_proj_text_emb, device=self.device)
                uma_loss, _, _ = clip_loss(agg_proj_ecg_emb1, agg_proj_ecg_emb2, device=self.device)
                loss = cma_loss + uma_loss

                # accumalate loss for logging
                val_cma_loss += cma_loss.item()
                val_uma_loss += uma_loss.item()
                val_loss += loss.item()
                val_epoch_acc1.append(acc1.item())
                val_epoch_acc5.append(acc5.item())
        
        if self.device == 0:
            val_cma_loss = val_cma_loss/len(val_epoch_acc1)
            val_uma_loss = val_uma_loss/len(val_epoch_acc1)
            val_loss = val_loss/len(val_epoch_acc1)
            val_epoch_acc1 = np.array(val_epoch_acc1).mean()
            val_epoch_acc5 = np.array(val_epoch_acc5).mean()
            
            val_log = {'val_loss': val_loss,
                        'val_cma_loss': val_cma_loss,
                        'val_uma_loss': val_uma_loss,
                        'val_acc1': val_epoch_acc1,
                        'val_acc5': val_epoch_acc5}
            return val_log
        else:
            return None
        
    def save_checkpoints(self, epoch, PATH):

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            PATH)
    