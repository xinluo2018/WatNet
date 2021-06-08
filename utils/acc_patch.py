import tensorflow as tf

class miou_binary(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.where(y_pred>0.5, 1, 0)
        super().update_state(y_true, y_pred, sample_weight)

