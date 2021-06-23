"""
References:
[1] https://www.fon.hum.uva.nl/praat/manual/Confusion__To_Similarity___.html
[2] Klein, Plomp, Pols (1970). Vowel spectra, vowel spaces, and vowel
    identification. Journal of the Acoustical Society of America.
"""

import numpy as np
from utils.paths import path_confusion, path_similarity_cnn

# The number of examples per class varies, so normalise the rows of `C`
C = np.loadtxt(path_confusion)
C = C / np.sum(C, axis=1, keepdims=True)
similarity_cnn = np.zeros((1000, 1000))

for i in range(1000):
    for j in range(1000):
        # Note that the factor of 1/2 is present in ref [2] but not ref [1]
        similarity_cnn[i, j] = (1/2) * np.sum(C[i] + C[j] - np.abs(C[i] - C[j]))

np.savetxt(path_similarity_cnn, similarity_cnn, fmt='%.18f')
