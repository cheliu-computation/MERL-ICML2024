import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score, f1_score
import yaml as yaml
import sys
sys.path.append("../finetune/")

from finetune_dataset import getdataset as get_zero_dataset

def compute_AUCs(gt, pred, n_class):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
        true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
        can either be probability estimates of the positive class,
        confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt
    pred_np = pred
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i], average='macro', multi_class='ovo'))
    return AUROCs

def get_class_emd(model, class_name, device='cuda'):
    model.eval()
    with torch.no_grad(): # to(device=torch.device("cuda"iftorch.cuda.is_available()else"cpu")) 
        zeroshot_weights = []
        # compute embedding through model for each class
        for texts in tqdm(class_name):
            texts = texts.lower()
            texts = [texts] # convert to list
            texts = model._tokenize(texts) # tokenize
            class_embeddings = model.get_text_emb(texts.input_ids.to(device=device)
                                                            , texts.attention_mask.to(device=device)
                                                            ) # embed with text encoder
            class_embeddings = model.proj_t(class_embeddings) # embed with text encoder

            # normalize class_embeddings
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # average over templates 
            class_embedding = class_embeddings.mean(dim=0) 
            # norm over new averaged templates
            class_embedding /= class_embedding.norm() 
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights

def get_ecg_emd(model, loader, zeroshot_weights, device='cuda', softmax_eval=True):
    y_pred = []
    model.eval()
    with torch.no_grad():
        for i, (ecg, target) in enumerate(tqdm(loader)):
            ecg = ecg.to(device=device) 
            # predict
            ecg_emb = model.ext_ecg_emb(ecg)
            ecg_emb /= ecg_emb.norm(dim=-1, keepdim=True)

            # obtain logits (cos similarity)
            logits = ecg_emb @ zeroshot_weights
            logits = torch.squeeze(logits, 0) # (N, num_classes)
            if softmax_eval is False: 
                norm_logits = (logits - logits.mean()) / (logits.std())
                logits = torch.sigmoid(norm_logits) 
            
            y_pred.append(logits.cpu().data.numpy())
        
    y_pred = np.concatenate(y_pred, axis=0)
    return np.array(y_pred)

def zeroshot_eval(model, set_name, device='cuda', args_zeroshot_eval=None):
    assert args_zeroshot_eval is not None, "Please specify the test set!"

    set_name = set_name
    num_workers = args_zeroshot_eval['num_workers']
    batch_size = args_zeroshot_eval['batch_size']

    meta_data_path = args_zeroshot_eval['meta_data_path']

    if 'val_sets' not in args_zeroshot_eval.keys():
        data_path = args_zeroshot_eval['test_sets'][set_name]['data_path']
    if 'val_sets' in args_zeroshot_eval.keys():
        data_path = args_zeroshot_eval['val_sets'][set_name]['data_path']
    
    data_path = os.path.join(meta_data_path, data_path)

    meta_split_path = args_zeroshot_eval['meta_split_path']
    if 'val_sets' not in args_zeroshot_eval.keys():
        split_path = args_zeroshot_eval['test_sets'][set_name]['split_path']
    if 'val_sets' in args_zeroshot_eval.keys():
        split_path = args_zeroshot_eval['val_sets'][set_name]['split_path']
    split_path = os.path.join(meta_split_path, split_path)


    if 'ptbxl' in set_name:
        test_dataset = get_zero_dataset(data_path, split_path, mode='test', dataset_name='ptbxl')
    else:
        test_dataset = get_zero_dataset(data_path, split_path, mode='test', dataset_name=set_name)
    class_name = test_dataset.labels_name

    # open json as dict
    with open(args_zeroshot_eval['prompt_dict'], 'r') as f:
        prompt_dict = yaml.load(f, Loader=yaml.FullLoader)

    # get prompt for each class
    target_class = [prompt_dict[i] for i in class_name]
    
    print('***********************************')
    print('zeroshot classification set is {}'.format(set_name))
    
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
        )

    # get the target array from testset
    gt = test_dataset.labels

    # get class embedding
    zeroshot_weights = get_class_emd(model.module, target_class, device=device)
    # get ecg prediction
    pred = get_ecg_emd(model.module, test_dataloader, 
                       zeroshot_weights, device=device, softmax_eval=True)
    
    AUROCs = compute_AUCs(gt, pred, len(target_class))
    AUROCs = [i*100 for i in AUROCs]
    AUROC_avg = np.array(AUROCs).mean()
    
    max_f1s = []
    accs = []
    
    for i in range(len(target_class)):   
        gt_np = gt[:, i]
        pred_np = pred[:, i]
        precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1 = np.max(f1_scores)
        max_f1_thresh = thresholds[np.argmax(f1_scores)]
        max_f1s.append(max_f1)
        accs.append(accuracy_score(gt_np, pred_np>max_f1_thresh))
    
    
    max_f1s = [i*100 for i in max_f1s]
    accs = [i*100 for i in accs]
    f1_avg = np.array(max_f1s).mean()    
    acc_avg = np.array(accs).mean()

    res_dict = {'AUROC_avg': AUROC_avg,
                'F1_avg': f1_avg,
                'ACC_avg': acc_avg
    }
    for i in range(len(target_class)):
        res_dict.update({f'AUROC_{class_name[i]}': AUROCs[i],
                        f'F1_{class_name[i]}': max_f1s[i],
                        f'ACC_{class_name[i]}': accs[i]
        })

    print('-----------------------------------')
    print('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=AUROC_avg))
    for i in range(len(target_class)):
        print('The AUROC of {} is {}'.format(class_name[i], AUROCs[i]))
        
    print('-----------------------------------')
    print('The average f1 is {F1_avg:.4f}'.format(F1_avg=f1_avg))
    for i in range(len(target_class)):
        print('The F1 of {} is {}'.format(class_name[i], max_f1s[i]))

    print('-----------------------------------')
    print('The average ACC is {ACC_avg:.4f}'.format(ACC_avg=acc_avg))
    for i in range(len(target_class)):
        print('The ACC of {} is {}'.format(class_name[i], accs[i]))
    print('***********************************')

    return f1_avg, acc_avg, AUROC_avg, max_f1s, accs, AUROCs, res_dict
