import os
import random
import yaml as yaml
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch

import sys
sys.path.append("../utils")
import utils_builder
from zeroshot_val import zeroshot_eval

os.environ["TOKENIZERS_PARALLELISM"] = "true"

device_id = 'cuda'

config = yaml.load(open("zeroshot_config.yaml", "r"), Loader=yaml.FullLoader)

torch.manual_seed(42)
random.seed(0)
np.random.seed(0)

model = utils_builder.ECGCLIP(config['network'])
ckpt = 'your_ckpt_path'
ckpt = torch.load(f'{ckpt}', map_location='cpu')
model.load_state_dict(ckpt)
model = model.to(device_id)
model = torch.nn.DataParallel(model)

args_zeroshot_eval = config['zeroshot']

avg_f1, avg_acc, avg_auc = 0, 0, 0
for set_name in args_zeroshot_eval['test_sets'].keys():

        f1, acc, auc, _, _, _, res_dict = \
        zeroshot_eval(model=model, 
        set_name=set_name, 
        device=device_id, 
        args_zeroshot_eval=args_zeroshot_eval)

        avg_f1 += f1
        avg_acc += acc
        avg_auc += auc

avg_f1 = avg_f1/len(args_zeroshot_eval['test_sets'].keys())
avg_acc = avg_acc/len(args_zeroshot_eval['test_sets'].keys())
avg_auc = avg_auc/len(args_zeroshot_eval['test_sets'].keys())