import tensorflow as tf
from utils.acc_patch import miou_binary
import math
## ---- root dir ---- ##
root = '/home/yons/Desktop/developer-luo/WatNet'   # local sever
# root = '/content/drive/My Drive/WatNet'    # colab

## ---- super-parameter for model training ---- ##
patch_size = 512
num_bands = 6
epochs = 200
lr = 0.002
batch_size = 4
buffer_size = 200
# size_tra_scene = 64
size_scene = 95
step_per_epoch = math.ceil(size_scene/batch_size)


## ---- configuration for model training ---- ##
class lr_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):    
  def __init__(self, initial_learning_rate, steps_all):
      self.initial_learning_rate = initial_learning_rate
      self.steps_all = steps_all
  def __call__(self, step):
     return self.initial_learning_rate*((1-step/self.steps_all)**0.9)
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=lr,
#     decay_steps=100,    # 1 step = 1 batch data
#     decay_rate=0.9)
loss_bce = tf.keras.losses.BinaryCrossentropy()
opt_adam = tf.keras.optimizers.Adam(learning_rate=\
                            lr_schedule(lr,step_per_epoch*epochs))

## ---- metrics ---- ##
tra_loss = tf.keras.metrics.Mean(name="tra_loss")
tra_oa = tf.keras.metrics.BinaryAccuracy('tra_oa')
tra_miou = miou_binary(num_classes=2, name='tra_miou')
val_loss = tf.keras.metrics.Mean(name="test_loss")
val_oa = tf.keras.metrics.BinaryAccuracy('test_oa')
val_miou = miou_binary(num_classes=2, name='test_miou')


