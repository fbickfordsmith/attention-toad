import os
import numpy as np
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
from .paths import path_conv5_ssd, path_images, path_wnids

def get_sorted_filepaths(data_partition, input_mode, wnid=None):
    if wnid == None:
        wnid = '*'
    if input_mode == 'conv5':
        extension = wnid + os.path.sep + '*.tfrecords'
        filepaths = (path_conv5_ssd/data_partition).glob(extension)
    else:
        extension = wnid + os.path.sep + '*.JPEG'
        filepaths = (path_images/data_partition).glob(extension)
    return sorted([str(filepath) for filepath in filepaths])

def load_image_tensorflow_hub(filepath):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.reshape(image, ((1,) + image.shape))

def prepare_dataset(filepaths, class_weights=None, cache=False):
    """
    If using cache for validation data, ensure ~12GB of memory is available.
    Applying cache after batching gives faster loading than the other way
    around. Use `tfds.core.benchmark` to assess dataset speed.
    """
    labels = [f.split(os.path.sep)[-2] for f in filepaths]
    if class_weights:
        sample_weights = np.array([class_weights[label] for label in labels])
    else:
        sample_weights = np.ones_like(labels, dtype=float)
    wnids = np.loadtxt(path_wnids, dtype=str)
    labels = np.array([(label == wnids) for label in labels])
    dataset_x = tf.data.Dataset.from_tensor_slices(filepaths)
    if '.tfrecords' in filepaths[0]:
        dataset_x = prepare_dataset_conv5(dataset_x)
    else:
        dataset_x = prepare_dataset_image(dataset_x)
    dataset_y = tf.data.Dataset.from_tensor_slices(labels)
    dataset_w = tf.data.Dataset.from_tensor_slices(sample_weights)
    dataset = tf.data.Dataset.zip((dataset_x, dataset_y, dataset_w))
    dataset = dataset.batch(128)
    if cache:
        dataset = dataset.cache()
    return dataset.prefetch(AUTOTUNE)

def prepare_dataset_conv5(dataset_x, shape_x=(7, 7, 512)):
    def parse(example):
        features = {'x':tf.io.FixedLenFeature((size_x,), tf.float32)}
        parsed_features = tf.io.parse_single_example(example, features)
        return tf.reshape(parsed_features['x'], shape_x)
    size_x = np.prod(shape_x)
    dataset_x = dataset_x.interleave(
        tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE)
    return dataset_x.map(parse, num_parallel_calls=AUTOTUNE)

def prepare_dataset_image(dataset_x, shape_x=(224, 224)):
    def load_image_numpy(filepath):
        filepath = filepath.numpy()
        x = tf.keras.preprocessing.image.load_img(filepath, target_size=shape_x)
        x = tf.keras.preprocessing.image.img_to_array(x)
        return tf.keras.applications.vgg16.preprocess_input(x)
    def load_image_tensorflow(filepath):
        x = tf.py_function(load_image_numpy, [filepath], tf.float32)
        return tf.reshape(x, shape_x + (3,))
    return dataset_x.map(load_image_tensorflow, num_parallel_calls=AUTOTUNE)

def save_tfrecord(x, filepath):
    x = x.flatten()
    feature = {'x':tf.train.Feature(float_list=tf.train.FloatList(value=x))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer = tf.io.TFRecordWriter(filepath)
    writer.write(example.SerializeToString())
    writer.close()
    return
