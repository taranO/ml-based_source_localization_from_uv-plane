import tensorflow as tf
from keras import backend as K

# ======================================================================================================================

class DistanceCorrelation(tf.keras.losses.Loss):
    def __init__(self):
        super(DistanceCorrelation, self).__init__()

    def __pairwise_dist(self, X):
        G = tf.matmul(X, tf.transpose(X))
        n = tf.shape(G)[0]
        m = tf.shape(G)[1]
        G_diag = tf.linalg.diag_part(G)
        d = tf.tile(tf.expand_dims(G_diag, axis=0), [n, 1]) - 2*G + tf.tile(tf.expand_dims(G_diag, axis=1), [1, m])
        d = K.clip(d, min_value=1e-8, max_value=K.max(d))

        return tf.sqrt(d)

    def call(self, X, Y):
        n = tf.shape(X)[0]
        m = tf.shape(Y)[0]
        X = tf.reshape(X, [-1, tf.shape(X)[1]*tf.shape(X)[2]])
        Y = tf.reshape(Y, [-1, tf.shape(Y)[1]*tf.shape(Y)[2]])

        a = self.__pairwise_dist(X)
        b = self.__pairwise_dist(Y)
        A = a - tf.squeeze(tf.reduce_mean(a, axis=1)) - tf.squeeze(tf.reduce_mean(a, axis=0)) + tf.squeeze(tf.reduce_mean(a))
        B = b - tf.squeeze(tf.reduce_mean(b, axis=1)) - tf.squeeze(tf.reduce_mean(b, axis=0)) + tf.squeeze(tf.reduce_mean(b))
        
        dcov = tf.reduce_sum(tf.math.multiply(A, B)) / tf.cast(n*m, tf.float32)
        dvarXX = tf.reduce_sum(tf.math.multiply(A, A)) / tf.cast(n**2, tf.float32)
        dvarYY = tf.reduce_sum(tf.math.multiply(B, B)) / tf.cast(m**2, tf.float32)

        dcorr = dcov / tf.sqrt(dvarXX * dvarYY)

        return -dcorr
