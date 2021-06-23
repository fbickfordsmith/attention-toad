import numpy as np
from .paths import (path_clutter, path_difficulty, path_names, path_scale,
    path_similarity_cnn, path_similarity_human, path_similarity_semantic,
    path_wnids)

try:
    clutter = np.loadtxt(path_clutter)
    difficulty = np.loadtxt(path_difficulty)
    scale = np.loadtxt(path_scale)
    similarity_cnn = np.loadtxt(path_similarity_cnn)
    similarity_human = np.loadtxt(path_similarity_human)
    similarity_semantic = np.loadtxt(path_similarity_semantic)
    names = np.loadtxt(path_names, dtype=str)
    wnids = np.loadtxt(path_wnids, dtype=str)
    property_names = ('Clutter', 'Difficulty', 'Scale', 'CNN similarity',
        'Human similarity', 'Semantic similarity')
except:
    pass

def compute_task_properties(i, j):
    clutter_ij = (clutter[i] + clutter[j]) / 2
    diff_ij = (difficulty[i] + difficulty[j]) / 2
    scale_ij = (scale[i] + scale[j]) / 2
    sim_cnn_ij = similarity_cnn[i, j]
    sim_hum_ij = similarity_human[i, j]
    sim_sem_ij = similarity_semantic[i, j]
    return clutter_ij, diff_ij, scale_ij, sim_cnn_ij, sim_hum_ij, sim_sem_ij

def convert_task(task, mode):
    """
    If `task.dtype == bool`, `task` is a 1000-dimensional binary vector.
    If `task.dtype == int`, `task` is a variable-length vector of class indices.
    If `task.dtype == str`, `task` is a variable-length vector of WNIDs
        (eg, `tasks == ['n02480495', ..., 'n03445777']`).
    """
    wnids = np.loadtxt(path_wnids, dtype=str)
    wnid2ind = {wnid:ind for ind, wnid in enumerate(wnids)}
    if task.dtype == 'bool' and mode == 'indices':
        assert len(task) == 1000
        return np.flatnonzero(task)
    elif task.dtype == 'bool' and mode == 'wnids':
        assert len(task) == 1000
        return wnids[np.flatnonzero(task)]
    elif task.dtype == 'int' and mode == 'binary':
        return np.array([(i in task) for i in range(1000)])
    elif task.dtype == 'int' and mode == 'wnids':
        return wnids[task]
    elif task.dtype == 'str' and mode == 'binary':
        return np.array([(wnid in task) for wnid in wnids])
    elif task.dtype == 'str' and mode == 'indices':
        return np.array([wnid2ind[wnid] for wnid in task])
    else:
        raise ValueError('Invalid combination of `task.dtype` and `mode`.')
