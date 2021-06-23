import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_tasks', type=int)
args = parser.parse_args()

import numpy as np
import scipy.spatial
import sobol_seq
from utils.paths import path_expt
from utils.tasks import compute_task_properties, convert_task

tasks = np.full((1, 1000), True)

if args.n_tasks > 0:
    tasks_all = [[i, j] for i in range(1000) for j in range(i+1, 1000)]
    tasks_all = np.array(tasks_all)
    properties_all = [compute_task_properties(i, j) for i, j in tasks_all]
    properties_all = np.array(properties_all)
    n_properties = properties_all.shape[1]
    targets = sobol_seq.i4_sobol_generate(dim_num=n_properties, n=args.n_tasks)
    for i in range(n_properties):
        # Scale dimension `i` of `targets` from the [0, 1] range to the range of
        # values of `properties_all` at dimension `i`
        low = np.min(properties_all[:, i])
        high = np.max(properties_all[:, i])
        width = high - low
        targets[:, i] = low + (targets[:, i] * width)
    distance = scipy.spatial.distance.cdist(targets, properties_all)
    inds_best, tasks_best = [], []
    for inds_sorted in np.argsort(distance, axis=1):
        i = 0
        while inds_sorted[i] in inds_best:
            i += 1
        inds_best.append(inds_sorted[i])
        tasks_best.append(convert_task(tasks_all[inds_sorted[i]], 'binary'))
    tasks_best = np.array(tasks_best)
    if len(tasks_best.shape) == 1:
        task_best = np.reshape(tasks_best, (1, -1))
    tasks = np.vstack((tasks, tasks_best))

np.savetxt(path_expt/'tasks.txt', tasks, fmt='%i')
