import tensorflow as tf
from keras import backend as K

# ======================================================================================================================
class DiceBCE(tf.keras.losses.Loss):
    def __init__(self):
        super(DiceBCE, self).__init__()

    def call(self, targets, inputs, smooth=1e-6):
        BCE = tf.keras.losses.binary_crossentropy(targets, inputs)

        intersection = K.sum(K.abs(targets * inputs), axis=-1)
        dice = 1 - (2. * intersection + smooth) / (K.sum(K.square(targets), -1) + K.sum(K.square(inputs), -1) + smooth)

        Dice_BCE = BCE + dice

        return Dice_BCE