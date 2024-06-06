import torch
from thop import profile
from thop import clever_format


import os
import numpy as np
import time
import sys
import argparse
import errno
from collections import OrderedDict
from tqdm import tqdm
import random

import torch
import torch.nn as nn

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.model_action import ActionNet

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-t', '--test', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=100)
    parser.add_argument('-dn', '--datanum', default=109)
    parser.add_argument('-kd', '--kidx', default=0)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    opts = parser.parse_args()
    return opts

def flops(args, opts):
    print(args)
    model_backbone = load_backbone(args)
    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    ske = torch.randn(1, 1, 64, 17, 3)
    macs, params = profile(model, inputs=((ske,)))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
        
if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    flops(args, opts)