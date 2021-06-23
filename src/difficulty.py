import numpy as np
from utils.paths import path_difficulty, path_expt

# The difficulty of a class is the error rate (= 1 - accuracy) of the baseline
# model on that class
difficulty = 1 - np.loadtxt(path_expt/'results.txt').reshape(-1, 1000)[0]
np.savetxt(path_difficulty, difficulty, fmt='%.18f')
