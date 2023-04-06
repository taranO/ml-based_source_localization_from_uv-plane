import tensorflow as tf
from tensorflow.keras import layers

from .base_model import BaseModel
from src.layers.weighted_sum import WeightedSum


# ======================================================================================================================
class InverseReIm(BaseModel):

    def __init__(self, config, args, size=128, name="Real_and_Imaginary"):
        super(InverseReIm, self).__init__(config, args)

        self._name = name
        self._size = size
        self.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    def __call__(self, input):
        x = input
        for i in range(len(self.config["dens_filters"])):
            x = layers.Dense(self.config["dens_filters"][i], use_bias=True, activation=layers.Activation("relu"))(x)

        x = layers.Reshape((2, self._size, self._size))(x)

        x = WeightedSum(size=self._size)([x[:, 0], x[:, 1]])
        x = layers.Activation("tanh")(x) # "sigmoid"

        return tf.keras.models.Model(input, x, name=self._name)


