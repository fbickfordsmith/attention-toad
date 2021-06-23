"""
References:
[1] stackoverflow.com/questions/46820500/how-to-handle-large-amouts-of-data-in-tensorflow
[2] stackoverflow.com/questions/47861084/how-to-store-numpy-arrays-as-tfrecord
[3] stackoverflow.com/questions/48889482/feeding-npy-numpy-files-into-tensorflow-data-pipeline
[4] gist.github.com/swyoon/8185b3dcf08ec728fb22b99016dd533f
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_partition', choices={'train', 'val', 'val_white'})
args = parser.parse_args()

import numpy as np
from tqdm import tqdm
from utils.data import get_sorted_filepaths, prepare_dataset, save_tfrecord
from utils.models import make_truncated_vgg16
from utils.paths import path_conv5_hdd, path_images, path_wnids

wnids = np.loadtxt(path_wnids, dtype=str)
vgg_trunc = make_truncated_vgg16()

for wnid in tqdm(wnids):
    (path_conv5_hdd/args.data_partition/wnid).mkdir(parents=True, exist_ok=True)
    filepaths_wnid = get_sorted_filepaths(args.data_partition, 'image', wnid)
    dataset_wnid = prepare_dataset(filepaths_wnid)
    conv5_wnid = vgg_trunc.predict(dataset_wnid)
    for i, filepath_i in enumerate(filepaths_wnid):
        filepath = filepath_i.replace(str(path_images), str(path_conv5_hdd))
        filepath = filepath.replace('JPEG', 'tfrecords')
        save_tfrecord(conv5_wnid[i], filepath)
