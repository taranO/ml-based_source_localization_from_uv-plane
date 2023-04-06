import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ======================================================================================================================
class WeightedSum(layers.Layer):
    def __init__( self, n_models = 2, size=128, **kwargs):
        super( WeightedSum, self ).__init__( **kwargs)
        self.n_models = n_models
        self.size = size

    def build(self,input_shape):
        self.sum_weights = self.add_weight(name="sum_weights",shape=(self.n_models, self.size, self.size),
                                    initializer = tf.keras.initializers.RandomUniform(0,1.0),
                                    constraint = NormalizeSumWeights(),
                                    trainable = True)

    def call(self,inputs):
        output = tf.multiply(tf.cast(self.sum_weights[0],inputs[0].dtype),inputs[0])
        for i in range(1,len(inputs)):
            output += tf.multiply(inputs[i],tf.cast(self.sum_weights[i], inputs[i].dtype))

        return output

    def get_config(self):
        data = { "n_models": self.n_models}
        return data
    
class WeightedSum1D(layers.Layer):
    def __init__( self, n_models = 2, size=128, **kwargs):
        super(WeightedSum1D, self ).__init__( **kwargs)
        self.n_models = n_models
        self.size = size

    def build(self,input_shape):
        self.sum_weights = self.add_weight(name="sum_weights", shape=(self.n_models, self.size),
                                    initializer = tf.keras.initializers.RandomUniform(0,1.0),
                                    constraint = NormalizeSumWeights(),
                                    trainable = True)

    def call(self,inputs):
        output = tf.multiply(tf.cast(self.sum_weights[0],inputs[0].dtype),inputs[0])
        for i in range(1,len(inputs)):
            output += tf.multiply(inputs[i],tf.cast(self.sum_weights[i], inputs[i].dtype))

        return output

    def get_config(self):
        data = { "n_models": self.n_models}
        return data    
    
class WeightedSum_v2(layers.Layer):
    def __init__( self, n_models = 2, size=128, **kwargs):
        super( WeightedSum, self ).__init__( **kwargs)
        self.n_models = n_models
        self.size = size

    def build(self,input_shape):
        self.sum_weights = self.add_weight(name="sum_weights",shape=(self.n_models, self.size, self.size),
                                    initializer = tf.keras.initializers.RandomUniform(0,1.0),
                                    constraint = NormalizeSumWeights(),
                                    trainable = True)

    def call(self, inputs):
        output = tf.multiply(tf.cast(self.sum_weights[0],inputs[:,0].dtype),inputs[:,0])
        # for i in range(1,len(inputs)):
        for i in range(1, inputs.shape[1]):
            output += tf.multiply(inputs[:,i],tf.cast(self.sum_weights[i], inputs[:,i].dtype))

        return output

    def get_config(self):
        data = { "n_models": self.n_models}
        return data

class NormalizeSumWeights(tf.keras.constraints.Constraint):
    def __init__(self,**kwargs):
        super(NormalizeSumWeights,self).__init__(**kwargs)

    def call(self,w):
        return tf.math.divide(w,tf.math.reduce_sum(w))
