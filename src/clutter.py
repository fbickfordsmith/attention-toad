import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.data import get_sorted_filepaths
from utils.paths import (path_clutter, path_cooccurrence, path_metadata,
    path_mids, path_objdet, path_wnids)

wnids = np.loadtxt(path_wnids, dtype=str)
mids = np.loadtxt(path_mids, dtype=str)
mid2ind = {mid:ind for ind, mid in enumerate(mids)}
cooccurrence = np.loadtxt(path_cooccurrence)
cooccurrence /= np.sum(cooccurrence, axis=1, keepdims=True) # make rows sum to 1
shorten_filepath = lambda filepath: '/'.join(filepath.split('/')[-2:])
clutter, filepaths = [], []

for i, wnid in enumerate(tqdm(wnids)):
    filepaths_wnid = get_sorted_filepaths('train', 'image', wnid)
    path_scores = path_objdet/f'{wnid}_scores.txt'
    path_class_names = path_objdet/f'{wnid}_class_names.txt'
    scores_wnid = np.loadtxt(path_scores)
    class_names_wnid = np.loadtxt(path_class_names, dtype=str)
    for scores, class_names in zip(scores_wnid, class_names_wnid):
        clutter_image = 0
        for class_name in np.unique(class_names):
            # If WNID `i` and MID `j` commonly cooccur, `cooccurrence[i, j]` is
            # high and MID `j` is relevant to WNID `i`
            j = mid2ind[class_name]
            irrelevance = 1 - cooccurrence[i, j]
            scores_j = scores[np.flatnonzero(class_names == class_name)]
            clutter_image += irrelevance * np.max(scores_j)
        clutter.append(clutter_image)
    filepaths.extend(filepaths_wnid)
    df_clutter = pd.DataFrame({'filepath':filepaths, 'clutter':clutter})
    df_clutter['filepath'] = df_clutter['filepath'].apply(shorten_filepath)
    df_clutter.to_csv(path_metadata/'imagenet_image_clutter.csv', index=False)

df_clutter['wnid'] = df_clutter['filepath'].str.split('_', expand=True)[0]
df_clutter_median = df_clutter.groupby('wnid').median()
df_clutter_median.reset_index(inplace=True)
np.savetxt(path_clutter, list(df_clutter_median['clutter']), fmt='%.18f')
