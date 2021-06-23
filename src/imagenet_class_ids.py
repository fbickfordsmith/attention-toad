import json
import numpy as np
from utils.paths import path_downloads, path_metadata, path_names, path_wnids

# WNID = machine identifier for an entity in the WordNet database
ind2wnidname = json.load(open(path_downloads/'imagenet_class_index.json'))
wnids, names = [], []

for i in range(1000):
    wnids.append(ind2wnidname[str(i)][0])
    names.append(ind2wnidname[str(i)][1])

np.savetxt(path_wnids, wnids, fmt='%s')
np.savetxt(path_names, names, fmt='%s')
