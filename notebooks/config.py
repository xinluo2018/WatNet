import tensorflow as tf
from utils.metrics import miou_binary

## -------- root dir -------- ##
root = '/home/yons/Desktop/developer-luo/WatNet'   # local sever
# root = '/content/drive/My Drive/WatNet'    # colab

## -------- super-parameter for model training -------- ##
patch_size = 512
epochs = 100
lr = 0.005
batch_size = 4
buffer_size = 200

## -------- configuration for model training -------- ##
loss_bce = tf.keras.losses.BinaryCrossentropy()
opt_adam = tf.keras.optimizers.Adam(lr)

## -------- metrics -------- ##
tra_loss = tf.keras.metrics.Mean(name="tra_loss")
tra_oa = tf.keras.metrics.BinaryAccuracy('tra_oa')
tra_miou = miou_binary(num_classes=2,name='tra_miou')
val_loss = tf.keras.metrics.Mean(name="test_loss")
val_oa = tf.keras.metrics.BinaryAccuracy('test_oa')
val_miou = miou_binary(num_classes=2,name='test_miou')


