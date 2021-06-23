import numpy as np
import tensorflow as tf
from .paths import path_expt, path_expt_raw

class ChannelwiseAttention(tf.keras.layers.Layer):
    """
    Multiply each channel of an input tensor by a separate attention weight.
    Reference: tensorflow.org/guide/keras/custom_layers_and_models.
    """
    def __init__(self, init='ones', **kwargs):
        super(ChannelwiseAttention, self).__init__()
        if 'task' in init:
            # Initialise with weights trained on a task (eg, `init='task0000'`)
            try:
                weights = np.loadtxt(path_expt/'weights.txt')
                if len(weights.shape) == 1:
                    # weights.shape: (n_weights,) -> (1, n_weights)
                    weights = np.reshape(weights, (1, -1))
                task_id = int(init[4:])
                weights = weights[task_id]
            except:
                weights = np.loadtxt(path_expt_raw/f'weights_{init}.txt')
            # weights.shape: (n_weights,) -> (1, 1, 1, n_weights)
            weights = np.reshape(weights, (1, 1, 1, -1))
            self.init = tf.constant_initializer(weights)
        else:
            self.init = init

    def build(self, input_shape):
        self.attention_weights = self.add_weight(
            name='attention_weights',
            shape=(1, 1, 1, input_shape[-1]),
            initializer=self.init,
            constraint=tf.keras.constraints.NonNeg())

    def call(self, x):
        return x * self.attention_weights

class SpatialAttention(tf.keras.layers.Layer):
    """
    Multiply each spatial component of an input tensor by a separate attention
    weight. Reference: tensorflow.org/guide/keras/custom_layers_and_models.
    """
    def __init__(self, init='ones'):
        super(SpatialAttention, self).__init__()
        if 'task' in init:
            # Initialise with weights trained on a task (eg, `init='task0000'`)
            try:
                weights = np.loadtxt(path_expt/'weights_spatial.txt')
                if len(weights.shape) == 1:
                    # weights.shape: (n_weights,) -> (1, n_weights)
                    weights = np.reshape(weights, (1, -1))
                task_id = int(init[4:])
                weights = weights[task_id]
            except:
                weights = np.loadtxt(path_expt_raw/f'weights_{init}_spatial.txt')
            # weights.shape: (n_weights,) -> (1, height, width, 1)
            height = width = int(np.sqrt(weights.shape[0]))
            weights = np.reshape(weights, (1, height, width, 1))
            self.init = tf.constant_initializer(weights)
        else:
            self.init = init

    def build(self, input_shape):
        self.attention_weights = self.add_weight(
            name='attention_weights',
            shape=(1, *input_shape[-3:-1], 1),
            initializer=self.init,
            constraint=tf.keras.constraints.NonNeg())

    def call(self, x):
        return x * self.attention_weights
