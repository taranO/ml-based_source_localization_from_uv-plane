from .base_model import BaseModel
from src.layers.weighted_mult import *


# ======================================================================================================================
class InverseMagPhas(BaseModel):

    def __init__(self, config, args, size=512, name="Magnitude_and_Phase"):
        super(InverseMagPhas, self).__init__(config, args)

        self._name = name
        self._size = size
        self.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001)  # 0.02

    def __call__(self, input):
        mag = input[:, 0]
        phas = input[:, 1]

        x = WeightedMult1D(size=input[:, 0].shape[2])([mag, phas])

        for i in range(len(self.config["dens_filters"])):
            x = layers.Dense(self.config["dens_filters"][i], use_bias=True)(x)
            x = layers.ReLU()(x)

        x = layers.Reshape((self._size, self._size))(x)

        return tf.keras.models.Model(input, x, name=self._name)
    
