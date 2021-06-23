import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int)
parser.add_argument('--attn_type', type=str, default='channelwise')
parser.add_argument('--init', type=str, default='task0000')
parser.add_argument('--intensity', type=float, default=0.5)
parser.add_argument('--eval_gap', type=int, default=20)
args = parser.parse_args()

import os
import numpy as np
from utils.layers import ChannelwiseAttention, SpatialAttention
from utils.models import make_attention_cnn
from utils.paths import path_expt
from utils.tasks import convert_task
from utils.testing import test
from utils.training import train

tasks_binary = np.loadtxt(path_expt/'tasks.txt', dtype=bool)
if len(tasks_binary.shape) == 1:
    tasks_binary = np.reshape(tasks_binary, (1, -1))

if args.attn_type == 'channelwise':
    attn_layer = ChannelwiseAttention(init=args.init)
    task_id = f'{args.task_id:04}'
else:
    attn_layer = SpatialAttention(init=args.init)
    task_id = f'{args.task_id:04}_spatial'

attn_cnn = make_attention_cnn(attn_layer, os.environ['INPUT_MODE'])

attn_cnn = train(
    model=attn_cnn,
    task_wnids=convert_task(tasks_binary[args.task_id], 'wnids'),
    task_id=task_id,
    intensity=args.intensity,
    input_mode=os.environ['INPUT_MODE'],
    eval_gap=args.eval_gap)

accuracies = test(
    model=attn_cnn,
    input_mode=os.environ['INPUT_MODE'],
    task_id_train=task_id)
