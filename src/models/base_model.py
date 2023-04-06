import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.models import Model


# ======================================================================================================================
class BaseModel(Model):

    def __init__(self, config, args):
        super(BaseModel, self).__init__()

        tf.keras.backend.set_image_data_format('channels_last')
        self.config = config
        self.args = args
        self.std = 0.02

    def residual_block(self, x, activation, kernel_size=(3, 3), strides=(1, 1), padding="valid", use_bias=False):

        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.std)
        gamma_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.std)
        dim = x.shape[-1]
        input_tensor = x

        x = ReflectionPadding2D()(input_tensor)
        x = layers.Conv2D(dim,
                          kernel_size,
                          strides=strides,
                          kernel_initializer=kernel_initializer,
                          padding=padding,
                          use_bias=use_bias,
                          )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        #x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        x = activation(x)

        x = ReflectionPadding2D()(x)
        x = layers.Conv2D(dim,
                          kernel_size,
                          strides=strides,
                          kernel_initializer=kernel_initializer,
                          padding=padding,
                          use_bias=use_bias,
                          )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        #x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        x = layers.add([input_tensor, x])

        return x

    def downsample(self, x, filters, activation, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False):
        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.std)
        gamma_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.std)

        x = layers.Conv2D(filters,
                          kernel_size,
                          strides=strides,
                          kernel_initializer=kernel_initializer,
                          padding=padding,
                          use_bias=use_bias,
                          )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        #x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        if activation:
            x = activation(x)
        return x

    def upsample(self, x, filters, activation, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False):
        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.std)
        gamma_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.std)

        x = layers.Conv2DTranspose(filters,
                                   kernel_size,
                                   strides=strides,
                                   padding=padding,
                                   kernel_initializer=kernel_initializer,
                                   use_bias=use_bias,
                                   )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        #x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        if activation:
            x = activation(x)
        return x


class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)
        # tf.keras.backend.set_image_data_format('channels_last')

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [

            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],

        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")
