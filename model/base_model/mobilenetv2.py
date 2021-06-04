import tensorflow as tf

##### MobileNetV2
relu6 = tf.keras.layers.ReLU(6.)
def _conv_block(inputs, filters, kernel, strides):
    x = tf.keras.layers.Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    return relu6(x)
def _bottleneck(inputs, filters, kernel, t, s, r=False):
    tchannel = inputs.shape[-1] * t
    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))
    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = relu6(x)
    x = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)  # 降维，改层为瓶颈层
    x = tf.keras.layers.BatchNormalization()(x)
    if r:
        x = tf.keras.layers.add([x, inputs])
    return x

def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    x = _bottleneck(inputs, filters, kernel, t, strides)
    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)
    return x

def MobileNetV2(input_shape, nclasses=2):
    
    """
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        classes: Integer, number of classes.
    # Returns
        MobileNetv2 model.
    """

    inputs = tf.keras.layers.Input(shape=input_shape, name='input')
    x = _conv_block(inputs, 32, (3, 3), strides=(2, 2))   # 0.5*size         n_layers = 1
 
    x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)  #        n_layers = 3
    x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)  # 0.5*size,  n_layers = 6

    x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)  # 0.5*size,  n_layers = 9
    x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)  # 0.5*size,  n_layers = 12
    x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)  #        n_layers = 9
    x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)  # 0.5*size,  n_layers = 9
    x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)  #        n_layers = 3

    x = _conv_block(x, 1280, (1, 1), strides=(1, 1))  # n_layers = 1
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Reshape((1, 1, 1280))(x)
    x = tf.keras.layers.Dropout(0.3, name='Dropout')(x)
    x = tf.keras.layers.Conv2D(nclasses, (1, 1), padding='same')(x)  # n_layers = 1
    x = tf.keras.layers.Activation('softmax', name='final_activation')(x)
    output = tf.keras.layers.Reshape((nclasses,), name='output')(x)
    model = tf.keras.models.Model(inputs, output)
    return model

# model = MobileNetV2(input_shape=(512,512,6), nclasses=2)
# model.summary()

