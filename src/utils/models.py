import tensorflow as tf

def get_compile_parameters(mixed_prec=False):
    opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
    if mixed_prec:
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    cce = tf.keras.metrics.CategoricalCrossentropy(name='loss_weighted_mean')
    return dict(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        weighted_metrics=[cce])

def make_attention_cnn(attn_layer, input_mode, mixed_prec=True):
    params_compile = get_compile_parameters(mixed_prec)
    vgg = tf.keras.applications.VGG16()
    name = f'{attn_layer.name}_cnn_{input_mode}'
    attn_cnn = tf.keras.models.Sequential(name=name)
    if input_mode == 'conv5':
        input_shape = vgg.get_layer('block5_pool').output_shape[1:]
        attn_cnn.add(tf.keras.layers.Input(shape=input_shape))
        if attn_layer:
            attn_cnn.add(attn_layer)
        adding = False
        for layer in vgg.layers:
            if adding:
                layer.trainable = False
                attn_cnn.add(layer)
            if layer.name == 'block5_pool':
                adding = True
    else:
        for layer in vgg.layers:
            layer.trainable = False
            attn_cnn.add(layer)
            if layer.name == 'block5_pool':
                attn_cnn.add(attn_layer)
    attn_cnn.compile(**params_compile)
    return attn_cnn

def make_truncated_vgg16(mixed_prec=False):
    params_compile = get_compile_parameters(mixed_prec)
    vgg = tf.keras.applications.VGG16()
    vgg_trunc = tf.keras.models.Sequential()
    for layer in vgg.layers:
        vgg_trunc.add(layer)
        if layer.name == 'block5_pool':
            break
    vgg_trunc.compile(**params_compile)
    return vgg_trunc
