import numpy as np
import pandas as pd
from utils.paths import path_downloads, path_metadata, path_mids

# MID = machine identifier for an entity in the Google Knowledge Graph
path_index = path_downloads/'class-descriptions-boxable.csv'
df = pd.read_csv(path_index, names=('mid', 'name'), header=None)
df['name'] = df['name'].str.replace(' ', '_')
np.savetxt(path_mids, df['mid'], fmt='%s')
np.savetxt(path_metadata/'openimages_class_names.txt', df['name'], fmt='%s')
