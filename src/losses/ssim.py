import tensorflow as tf

class SSIM(tf.keras.losses.Loss):
    def __init__(self):
        super(SSIM, self).__init__()

    def call(self, X, Y):
        return 1 - tf.reduce_mean(tf.image.ssim(X, Y, 1.0))
