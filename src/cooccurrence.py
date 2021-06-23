import numpy as np
from tqdm import tqdm
from utils.paths import path_cooccurrence, path_mids, path_objdet, path_wnids

wnids = np.loadtxt(path_wnids, dtype=str)
mids = np.loadtxt(path_mids, dtype=str)
mid2ind = {mid:ind for ind, mid in enumerate(mids)}
cooccurrence = np.zeros((1000, 601))

# `cooccurrence[i, j]` captures how commonly ImageNet class `i` cooccurs with
# OpenImages class `j`
for i, wnid in enumerate(tqdm(wnids)):
    path_scores = path_objdet/f'{wnid}_scores.txt'
    path_class_names = path_objdet/f'{wnid}_class_names.txt'
    scores_wnid = np.loadtxt(path_scores)
    class_names_wnid = np.loadtxt(path_class_names, dtype=str)
    for scores, class_names in zip(scores_wnid, class_names_wnid):
        for class_name in np.unique(class_names):
            j = mid2ind[class_name]
            scores_j = scores[np.flatnonzero(class_names == class_name)]
            cooccurrence[i, j] += np.max(scores_j)

np.savetxt(path_cooccurrence, cooccurrence, fmt='%.18f')
