import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--attn_type', type=str, default='channelwise')
args = parser.parse_args()

import pandas as pd
import numpy as np
from utils.paths import path_expt, path_expt_raw

n_all = len(list(path_expt_raw.glob('results_task*.txt')))
n_spatial = len(list(path_expt_raw.glob('results_task*_spatial.txt')))
n_tasks = n_spatial if args.attn_type == 'spatial' else n_all - n_spatial
training, weights, results = [], [], []

for i in range(n_tasks):
    task_id = f'task{i:04}_spatial' if args.attn_type == 'spatial' else f'task{i:04}'
    training_i = pd.read_csv(path_expt_raw/f'training_{task_id}.csv')
    weights_i = np.loadtxt(path_expt_raw/f'weights_{task_id}.txt')
    results_i = np.loadtxt(path_expt_raw/f'results_{task_id}.txt')
    training_i.insert(0, 'task', i)
    training.append(training_i)
    weights.append(np.reshape(weights_i, (1, -1)))
    results.append(np.reshape(results_i, (1, -1)))

training = pd.concat(training)
weights = np.vstack(weights)
results = np.vstack(results)
extension = '_spatial' if args.attn_type == 'spatial' else ''
training.to_csv(path_expt/f'training{extension}.csv', index=False)
np.savetxt(path_expt/f'weights{extension}.txt', weights, fmt='%.18f')
np.savetxt(path_expt/f'results{extension}.txt', results, fmt='%.18f')
