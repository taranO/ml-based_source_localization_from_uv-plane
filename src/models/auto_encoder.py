import tensorflow as tf
from tensorflow.keras import layers
from .base_model import BaseModel

# ======================================================================================================================

class AutoEncoder(BaseModel):

    def __init__(self, config, args, size=512, name="AE_QE"):
        super(AutoEncoder, self).__init__(config, args)

        self._name = name
        self._size = size

    def __call__(self, input):
        x = input
        x = layers.Reshape((self._size, self._size, 1))(x)

        for filter in self.config["filters"]:
            x = layers.ZeroPadding2D(padding=(1, 1))(x)
            x = layers.Conv2D(filter, (3, 3), strides=2, activation="relu", padding="valid")(x)
            x = layers.BatchNormalization()(x)

        for filter in self.config["filters"][::-1][1:]:
            x = layers.Conv2DTranspose(filter, (3, 3), strides=2, activation="relu", padding="same")(x)
            x = layers.BatchNormalization()(x)

        x = layers.Conv2DTranspose(1, (3, 3), strides=2, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("sigmoid")(x)
        
        return tf.keras.models.Model(input, x, name=self._name)
