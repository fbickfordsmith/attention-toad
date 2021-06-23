import os
import numpy as np
import tensorflow as tf
from .data import get_sorted_filepaths, prepare_dataset
from .paths import path_images, path_expt_raw, path_wnids

def test(model, input_mode, task_id_train=None):
    if input_mode == 'image_generator':
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input)
        x = datagen.flow_from_directory(
            path_images/'val_white', target_size=(224, 224), shuffle=False)
        filepaths = x.filepaths
    else:
        filepaths = get_sorted_filepaths('val_white', input_mode)
        x = prepare_dataset(filepaths)
    predictions_probability = model.predict(x, verbose=False)
    predictions_label = np.argmax(predictions_probability, axis=1)
    labels = np.array([f.split(os.path.sep)[-2] for f in filepaths])
    wnids = np.loadtxt(path_wnids, dtype=str)
    accuracies = []
    for i, wnid in enumerate(wnids):
        inds_wnid = np.flatnonzero(labels == wnid)
        accuracies.append(np.mean(predictions_label[inds_wnid] == i))
    if task_id_train:
        path_results = path_expt_raw/f'results_task{task_id_train}.txt'
        np.savetxt(path_results, accuracies, fmt='%.18f')
    return accuracies
