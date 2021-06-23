import numpy as np
from utils.paths import path_downloads, path_metadata

similarity_human = np.loadtxt(path_downloads/'marg_smat_4-221-6.txt')
path_similarity_human = path_metadata/'imagenet_class_similarity_human.txt'
np.savetxt(path_similarity_human, similarity_human, fmt='%.18f')
