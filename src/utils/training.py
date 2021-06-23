import copy
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from .data import get_sorted_filepaths, prepare_dataset
from .paths import path_expt_raw, path_wnids

wnids = np.loadtxt(path_wnids, dtype=str)

def train(model, task_wnids, task_id, intensity, input_mode, eval_gap):
    filepaths_train, filepaths_valid = get_train_valid_filepaths(input_mode)
    filepaths_in, filepaths_out = split_filepaths(filepaths_train, task_wnids)
    class_weights = weight_classes(filepaths_valid, task_wnids, intensity)
    dataset_valid = prepare_dataset(filepaths_valid, class_weights, True)
    np.random.seed(0)
    loss_best = np.inf
    metrics = {
        'epoch':[], 'loss_train':[], 'acc_train':[], 'loss_weighted_train':[],
        'loss_valid':[], 'acc_valid':[], 'loss_weighted_valid':[]}
    for i in tqdm(range(1000)):
        filepaths_i = sample_filepaths(filepaths_in, filepaths_out, intensity)
        dataset_i = prepare_dataset(filepaths_i)
        fit = model.fit(dataset_i, verbose=False, shuffle=False)
        save_weights(model, task_id, (i > 0))
        if i % eval_gap == 0:
            eval = model.evaluate(dataset_valid, verbose=False)
            metrics = update_save_metrics(metrics, i, fit, eval, task_id)
            loss_current = metrics['loss_weighted_valid'][-1]
            if loss_current <= loss_best:
                loss_best = loss_current
            else:
                # Backtrack to find the best weights
                loss_best = np.inf
                for j in range(i-eval_gap, i):
                    model = load_weights(model, task_id, j)
                    eval = model.evaluate(dataset_valid, verbose=False)
                    _, _, loss_current = eval
                    if loss_current <= loss_best:
                        loss_best = loss_current
                        j_best = j
                model = load_weights(model, task_id, j_best)
                save_weights(model, task_id, False)
                break
    print(f'Best loss: {loss_best:.6f} at epoch {j_best}.')
    return model

def get_train_valid_filepaths(input_mode, split=0.1):
    """For each class, reserve the first 10% of examples for validation."""
    filepaths_train, filepaths_valid = [], []
    for wnid in wnids:
        filepaths_wnid = get_sorted_filepaths('train', input_mode, wnid)
        n_valid = int(split * len(filepaths_wnid))
        filepaths_train.extend(filepaths_wnid[n_valid:])
        filepaths_valid.extend(filepaths_wnid[:n_valid])
    return filepaths_train, filepaths_valid

def split_filepaths(filepaths, task_wnids):
    filepaths = np.array(filepaths)
    labels = np.array([f.split(os.path.sep)[-2] for f in filepaths])
    filepaths_in, filepaths_out = [], []
    for wnid in wnids:
        filepaths_wnid = list(filepaths[np.flatnonzero(labels == wnid)])
        if wnid in task_wnids:
            filepaths_in.extend(filepaths_wnid)
        else:
            filepaths_out.append(filepaths_wnid)
    return filepaths_in, filepaths_out

def weight_classes(filepaths, task_wnids, intensity):
    """
    Check that `sum_in == sum_out`:
    ```
    sum_in, sum_out = 0, 0
    for wnid in wnids:
        n_examples = np.count_nonzero(labels == wnid)
        if wnid in task_wnids:
            sum_in += class_weights[wnid] * n_examples
        else:
            sum_out += class_weights[wnid] * n_examples
    print(sum_in, sum_out)
    ```
    """
    labels = np.array([f.split(os.path.sep)[-2] for f in filepaths])
    n_examples_in = sum(np.count_nonzero(labels == w) for w in task_wnids)
    n_examples_out = len(filepaths) - n_examples_in
    class_weights = {}
    for wnid in wnids:
        if wnid in task_wnids:
            class_weights[wnid] = intensity / n_examples_in
        else:
            class_weights[wnid] = (1 - intensity) / n_examples_out
    return class_weights

def sample_filepaths(filepaths_in, filepaths_out, intensity):
    """
    `filepaths_in` is a list of filepaths for in-set examples.
    `filepaths_out` is a list of lists, where `filepaths_out[i]` is a list of
        filepaths corresponding to the ith out-of-set class.
    `intensity` is the number of in-set examples as a proportion of the total
        number of examples: `intensity = N_in / (N_in + N_out)`. We can
        rearrange this to get `N_out = N_in * ((1 / intensity) - 1)`, which we
        use to set `n_left_to_sample`. An intensity of 0.5 gives `N_in = N_out`.
    """
    filepaths_out_copy = copy.deepcopy(filepaths_out)
    filepaths_out_sampled = []
    inds_to_sample_from = range(len(filepaths_out))
    n_left_to_sample = int(len(filepaths_in) * ((1 / intensity) - 1))
    while n_left_to_sample > 0:
        if n_left_to_sample < len(filepaths_out):
            inds_to_sample_from = np.random.choice(
                inds_to_sample_from, n_left_to_sample, replace=False)
        for i in inds_to_sample_from:
            sample = np.random.choice(filepaths_out_copy[i])
            filepaths_out_copy[i].remove(sample)
            filepaths_out_sampled.append(sample)
        n_left_to_sample -= len(inds_to_sample_from)
    return np.random.permutation(filepaths_in + filepaths_out_sampled)

def update_save_metrics(metrics, epoch, fit, eval, task_id):
    metrics['epoch'].append(epoch)
    metrics['loss_train'].append(fit.history['loss'][0])
    metrics['acc_train'].append(fit.history['accuracy'][0])
    metrics['loss_weighted_train'].append(fit.history['loss_weighted_mean'][0])
    metrics['loss_valid'].append(eval[0])
    metrics['acc_valid'].append(eval[1])
    metrics['loss_weighted_valid'].append(eval[2])
    path_metrics = path_expt_raw/f'training_task{task_id}.csv'
    pd.DataFrame(metrics).to_csv(path_metrics, index=False)
    return metrics

def load_weights(model, task_id, epoch):
    path_weights = path_expt_raw/f'weights_task{task_id}.txt'
    weights = np.loadtxt(path_weights)[epoch]
    if 'spatial' in task_id:
        # weights.shape: (n_weights,) -> (1, height, width, 1)
        height = width = int(np.sqrt(weights.shape[0]))
        weights = np.reshape(weights, (1, height, width, 1))
        model.get_layer('spatial_attention').set_weights([weights])
    else:
        # weights.shape: (n_weights,) -> (1, 1, 1, n_weights)
        weights = np.reshape(weights, (1, 1, 1, -1))
        model.get_layer('channelwise_attention').set_weights([weights])
    return model

def save_weights(model, task_id, append):
    path_weights = path_expt_raw/f'weights_task{task_id}.txt'
    if 'spatial' in task_id:
        weights = model.get_layer('spatial_attention').get_weights()[0]
    else:
        weights = model.get_layer('channelwise_attention').get_weights()[0]
    # weights.shape: (1, height, width, 1) or (1, 1, 1, n_weights) -> (1, n_weights)
    weights = np.reshape(weights, (1, -1))
    if append:
        try:
            weights_past = np.loadtxt(path_weights)
            if len(weights_past.shape) == 1:
                # weights_past.shape: (n_weights,) -> (1, n_weights)
                weights_past = np.reshape(weights_past, (1, -1))
            # weights.shape == (n_past_saves+1, n_weights)
            weights = np.vstack((weights_past, weights))
        except:
            pass
    np.savetxt(path_weights, weights, fmt='%.18f')
    return
