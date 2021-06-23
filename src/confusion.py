import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.data import get_sorted_filepaths, prepare_dataset
from utils.layers import ChannelwiseAttention
from utils.models import make_attention_cnn
from utils.paths import path_confusion, path_wnids

wnids = np.loadtxt(path_wnids, dtype=str)
attn_layer = ChannelwiseAttention(init='task0000')
input_mode = os.environ['INPUT_MODE']
attn_cnn = make_attention_cnn(attn_layer, input_mode)
labels, predictions = [], []

for i, wnid in enumerate(tqdm(wnids)):
    filepaths_wnid = get_sorted_filepaths('val_white', input_mode, wnid)
    dataset_wnid = prepare_dataset(filepaths_wnid)
    predictions_wnid = attn_cnn.predict(dataset_wnid)
    labels.extend(len(filepaths_wnid) * [i])
    predictions.extend(list(np.argmax(predictions_wnid, axis=1)))

# `C[i, j]` corresponds to how many times the predicted image label is `j` when
# the true label is `i`
C = tf.math.confusion_matrix(labels, predictions, num_classes=1000)
np.savetxt(path_confusion, np.array(C, dtype=np.int16), fmt='%i')
