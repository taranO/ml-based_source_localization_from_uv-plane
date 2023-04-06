import tensorflow as tf
from keras import backend as K

# ======================================================================================================================

class DiceCoefficient(tf.keras.losses.Loss):
    def __init__(self):
        super(DiceCoefficient, self).__init__()

    def call(self, y_true, y_pred, smooth=100):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        dice = (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

        return 1 - dice