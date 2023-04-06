import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ======================================================================================================================
class WeightedMult(layers.Layer):
    def __init__(self, n_models=2, size=128, **kwargs):
        super(WeightedMult, self).__init__(**kwargs)
        self.n_models = n_models
        self.size = size

    def build(self, input_shape):
        self.mult_weights = self.add_weight(name="mult_weights", shape=(self.n_models, self.size, self.size),
                                            initializer=tf.keras.initializers.RandomUniform(0, 1.0),
                                            constraint=NormalizeWeights(),
                                            trainable=True)

    def call(self, inputs):
        output = tf.multiply(tf.cast(self.mult_weights[0], inputs[0].dtype), inputs[0])
        for i in range(1, len(inputs)):
            output *= tf.multiply(inputs[i], tf.cast(self.mult_weights[i], inputs[i].dtype))

        return output

    def get_config(self):
        data = {"n_models": self.n_models}
        return data


class WeightedMult1D(layers.Layer):
    def __init__(self, n_models=2, size=128, **kwargs):
        super(WeightedMult1D, self).__init__(**kwargs)
        self.n_models = n_models
        self.size = size

    def build(self, input_shape):
        self.mult_weights = self.add_weight(name="mult_weights", shape=(self.n_models, self.size),
                                            initializer=tf.keras.initializers.RandomUniform(0, 1.0),
                                            constraint=NormalizeWeights(),
                                            trainable=True)
    def call(self, inputs):

        output = tf.multiply(tf.cast(self.mult_weights[0], inputs[0].dtype), inputs[0])
        for i in range(1, len(inputs)):
            output *= tf.multiply(inputs[i], tf.cast(self.mult_weights[i], inputs[i].dtype))

        return output

    def get_config(self):
        data = {"n_models": self.n_models}
        return data


class NormalizeWeights(tf.keras.constraints.Constraint):
    def __init__(self, **kwargs):
        super(NormalizeWeights, self).__init__(**kwargs)

    def call(self, w):
        return tf.math.divide(w, tf.math.reduce_sum(w))
