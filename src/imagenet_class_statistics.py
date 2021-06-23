import numpy as np
import pandas as pd

from utils.paths import (path_clutter, path_difficulty, path_metadata,
    path_names, path_scale, path_similarity_cnn, path_similarity_human,
    path_similarity_semantic, path_wnids)

clutter = np.loadtxt(path_clutter)
difficulty = np.loadtxt(path_difficulty)
scale = np.loadtxt(path_scale)
similarity_cnn = np.loadtxt(path_similarity_cnn)
similarity_human = np.loadtxt(path_similarity_human)
similarity_semantic = np.loadtxt(path_similarity_semantic)
names = np.loadtxt(path_names, dtype=str)
wnids = np.loadtxt(path_wnids, dtype=str)

inds_similar_cnn = np.argsort(similarity_cnn, axis=1)[:, ::-1][:, :5]
inds_similar_human = np.argsort(similarity_human, axis=1)[:, ::-1][:, :5]
inds_similar_semantic = np.argsort(similarity_semantic, axis=1)[:, ::-1][:, :5]

similar_cnn = [', '.join(n) for n in names[inds_similar_cnn]]
similar_human = [', '.join(n) for n in names[inds_similar_human]]
similar_semantic = [', '.join(n) for n in names[inds_similar_semantic]]

df = pd.DataFrame({
    'WordNet ID':wnids,
    'Name':names,
    'Clutter':clutter,
    'Difficulty':difficulty,
    'Scale':scale,
    'CNN similarity: highest ranked':similar_cnn,
    'Human similarity: highest ranked':similar_human,
    'Semantic similarity: highest ranked':similar_semantic})

path_statistics = path_metadata/'imagenet_class_statistics.csv'
df.to_csv(path_statistics, index=False, float_format='%.4f')
