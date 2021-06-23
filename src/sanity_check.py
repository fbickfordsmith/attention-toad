import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--verbose', default=False, action='store_true')
args = parser.parse_args()

import numpy as np
import tensorflow as tf
from utils.layers import ChannelwiseAttention, SpatialAttention
from utils.models import make_attention_cnn
from utils.testing import test

vgg = tf.keras.applications.VGG16()
vgg.compile(loss='categorical_crossentropy', metrics=['acc'])

# Mixed precision needs to be off to match VGG16
attn_layer_chan = ChannelwiseAttention(init='ones')
attn_layer_spat = SpatialAttention(init='ones')
attn_cnn_channel_conv5 = make_attention_cnn(attn_layer_chan, 'conv5', False)
attn_cnn_channel_image = make_attention_cnn(attn_layer_chan, 'image', False)
attn_cnn_spatial_conv5 = make_attention_cnn(attn_layer_spat, 'conv5', False)
attn_cnn_spatial_image = make_attention_cnn(attn_layer_spat, 'image', False)

models = (
    vgg,
    attn_cnn_channel_conv5, attn_cnn_channel_image,
    attn_cnn_spatial_conv5, attn_cnn_spatial_image)

if args.verbose:
    for model in models:
        print(model.summary(), '\n')

accuracies = (
    test(vgg, 'image_generator'), test(vgg, 'image'),
    test(attn_cnn_channel_conv5, 'conv5'), test(attn_cnn_channel_image, 'image'),
    test(attn_cnn_spatial_conv5, 'conv5'), test(attn_cnn_spatial_image, 'image'))

close = [np.allclose(accuracies[0], accuracy) for accuracy in accuracies[1:]]
result = 'passed' if np.all(close) else 'failed'
print(f'\nSanity check {result}.')
