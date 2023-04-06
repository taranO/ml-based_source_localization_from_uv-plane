import tensorflow as tf
from keras import backend as K

# ======================================================================================================================
class JaccardDistance(tf.keras.losses.Loss):
    def __init__(self):
        super(JaccardDistance, self).__init__()

    def call(self, y_true, y_pred, smooth=100):
        '''
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
        # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
        # Returns
        The Jaccard distance between the two tensors.
        # References
        - https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
        - http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf
        '''

        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)

        jac = (1 - jac) * smooth

        return jac

