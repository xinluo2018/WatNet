import random
import tensorflow as tf

def img_aug(image, truth, flip = True, rot = True, noisy = True):
    '''Data augmentation: noisy, filp, rotate. '''
    # image = tf.convert_to_tensor(image, dtype=tf.float32)
    # truth = tf.convert_to_tensor(truth, dtype=tf.float32)    
    if len(truth.shape) == 2:
        truth = tf.expand_dims(truth, axis=-1)
    if flip == True:
        if tf.random.uniform(()) > 0.5:
            if random.randint(1,2) == 1:  ## horizontal or vertical mirroring
                image = tf.image.flip_left_right(image)
                truth = tf.image.flip_left_right(truth)
            else: 
                image = tf.image.flip_up_down(image)
                truth = tf.image.flip_up_down(truth)
    if rot == True:
        if tf.random.uniform(()) > 0.5: 
            degree = random.randint(1,3)
            image = tf.image.rot90(image, k=degree)
            truth = tf.image.rot90(truth, k=degree)
    if noisy == True:
        if tf.random.uniform(()) > 0.5:
            std = random.uniform(0.001, 0.05)
            gnoise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=std, dtype=tf.float32)
            image = tf.add(image, gnoise)
    return image, truth


