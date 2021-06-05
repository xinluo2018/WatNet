
"""
@Reference: https://github.com/luyanger1799/amazing-semantic-segmentation
"""
import os
import sys
sys.path.append(os.getcwd())
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.backend as backend
from model.base_model.xception import Xception

class GlobalAveragePooling2D(layers.GlobalAveragePooling2D):
    def __init__(self, keep_dims=False, **kwargs):
        super(GlobalAveragePooling2D, self).__init__(**kwargs)
        self.keep_dims = keep_dims
    def call(self, inputs):
        if self.keep_dims is False:
            return super(GlobalAveragePooling2D, self).call(inputs)
        else:
            return backend.mean(inputs, axis=[1, 2], keepdims=True)

class Concatenate(layers.Concatenate):
    def __init__(self, out_size=None, axis=-1, name=None):
        super(Concatenate, self).__init__(axis=axis, name=name)
        self.out_size = out_size
    def call(self, inputs):
        return backend.concatenate(inputs, self.axis)

class deeplabv3_plus(tf.keras.Model):
    def __init__(self, nclasses, base_model='Xception-DeepLab', **kwargs):
        super(deeplabv3_plus, self).__init__(**kwargs)
        """
        The initialization of DeepLabV3Plus.
        :param num_classes: the number of predicted classes.
        :param version: 'DeepLabV3Plus'
        :param base_model: the backbone model
        :param kwargs: other parameters
        """
        dilation = [1, 2]
        self.base_model = base_model
        self.nclasses = nclasses
        self.dilation = dilation
        self.encoder = Xception(version=base_model, dilation=dilation)

    def call(self, inputs):
        nclasses = self.nclasses
        _, h, w, _ = backend.int_shape(inputs)
        self.aspp_size = (h // 16, w // 16)
        c2, c5 = self.encoder(inputs, output_stages=['c1', 'c5'])

        x = self._aspp(c5, 256)
        x = layers.Dropout(rate=0.5)(x)

        x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
        x = self._conv_bn_relu(x, 48, 1, strides=1)

        x = Concatenate(out_size=self.aspp_size)([x, c2])
        x = self._conv_bn_relu(x, 256, 3, 1)
        x = layers.Dropout(rate=0.5)(x)

        x = self._conv_bn_relu(x, 256, 3, 1)
        x = layers.Dropout(rate=0.1)(x)
        if nclasses == 2:
            x = layers.Conv2D(1, 1, strides=1, activation= 'sigmoid')(x)
        else:
            x = layers.Conv2D(nclasses, 1, strides=1, activation='softmax')(x)
        x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
        outputs = x
        return outputs
        # return models.Model(inputs, outputs, name='deeplabv3_plus')

    def _conv_bn_relu(self, x, filters, kernel_size, strides=1):
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def _aspp(self, x, out_filters):
        xs = list()
        x1 = layers.Conv2D(out_filters, 1, strides=1)(x)
        xs.append(x1)

        for i in range(3):
            xi = layers.Conv2D(out_filters, 3, strides=1, padding='same', dilation_rate=6 * (i + 1))(x)
            xs.append(xi)
        img_pool = GlobalAveragePooling2D(keep_dims=True)(x)
        img_pool = layers.Conv2D(out_filters, 1, 1, kernel_initializer='he_normal')(img_pool)
        img_pool = layers.UpSampling2D(size=self.aspp_size, interpolation='bilinear')(img_pool)
        xs.append(img_pool)

        x = Concatenate(out_size=self.aspp_size)(xs)
        x = layers.Conv2D(out_filters, 1, strides=1, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        return x

input = tf.ones([4, 256, 256, 4],tf.float32)
# input = layers.Input(shape=(256,256,3))
model = deeplabv3_plus(nclasses=2)
oupt = model(inputs=input)
print(oupt.shape)