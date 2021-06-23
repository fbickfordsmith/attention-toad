import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--class_id', type=int, choices=range(1000))
args = parser.parse_args()

import numpy as np
import tensorflow_hub
from tqdm import tqdm
from utils.data import get_sorted_filepaths, load_image_tensorflow_hub
from utils.paths import path_objdet, path_wnids

# `object_detector` is a FasterRCNN+InceptionResNetV2 trained on Open Images V4
url = 'https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1'
object_detector = tensorflow_hub.load(url).signatures['default']
wnids = np.loadtxt(path_wnids, dtype=str)
wnid = wnids[args.class_id]
filepaths = get_sorted_filepaths('train', 'image', wnid)
scores, class_names = [], []

for filepath in tqdm(filepaths):
    image = load_image_tensorflow_hub(filepath)
    predictions = object_detector(image)
    # `scores` are probabilities; `class_names` are OpenImages MIDs
    scores.append(predictions['detection_scores'].numpy())
    class_names.append(predictions['detection_class_names'].numpy())

scores, class_names = np.stack(scores), np.stack(class_names).astype(str)
np.savetxt(path_objdet/f'{wnid}_scores.txt', scores, fmt='%.18f')
np.savetxt(path_objdet/f'{wnid}_class_names.txt', class_names, fmt='%s')
