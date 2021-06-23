import numpy as np
import pandas as pd
from utils.paths import (path_downloads, path_metadata,
    path_similarity_semantic, path_wnids)

wnids = np.loadtxt(path_wnids, dtype=str)
path_pickle = path_downloads/'imagenet_mintree.unitsphere.pickle'
data_pickle = pd.read_pickle(path_pickle)
embeddings_pickle = data_pickle['embedding']
wnid2ind_pickle = data_pickle['label2ind']
embeddings = []

for wnid in wnids:
    ind_pickle = wnid2ind_pickle[wnid]
    embeddings.append(embeddings_pickle[ind_pickle])

# `similarity_semantic[i, j] == np.dot(embeddings[i], embeddings[j])`
embeddings = np.array(embeddings)
similarity_semantic = np.matmul(embeddings, embeddings.T)
path_embeddings = path_metadata/'imagenet_class_embeddings.txt'
np.savetxt(path_embeddings, embeddings, fmt='%.18f')
np.savetxt(path_similarity_semantic, similarity_semantic, fmt='%.18f')
