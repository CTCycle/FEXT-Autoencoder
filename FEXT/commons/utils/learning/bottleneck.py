import keras
from keras import activations, layers
import torch

from FEXT.commons.constants import CONFIG

   
    
# [BOTTLENECK DECOMPRESSION AND RESHAPING]
###############################################################################
@keras.utils.register_keras_serializable(package='CustomLayers', name='CompressionLayer')
class CompressionLayer(layers.Layer):
    
    def __init__(self, units, dropout_rate=0.2, **kwargs):
        super(CompressionLayer, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.dense1 = layers.Dense(
            units, kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l2(1e-5))
        self.dense2 = layers.Dense(
            units, kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l2(1e-5))
        self.batch_norm1 = layers.BatchNormalization()
        self.batch_norm2 = layers.BatchNormalization()
        self.dropout = layers.Dropout(dropout_rate)

    #--------------------------------------------------------------------------
    def call(self, inputs, training=None):
        batch_size, height, width, channels = keras.ops.shape(inputs)
        sequence_dim = height * width
        reshaped = keras.ops.reshape(inputs, (batch_size, sequence_dim, channels))
        layer = self.dense1(reshaped)
        layer = self.batch_norm1(layer, training=training)
        layer = keras.activations.relu(layer)
        layer = self.dropout(layer, training=training)
        layer = self.dense2(layer)
        layer = self.batch_norm2(layer, training=training)
        output = keras.activations.relu(layer)
        return output

    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(CompressionLayer, self).get_config()
        config.update({'units': self.units,
                       'dropout_rate': self.dropout_rate})
        return config

    # deserialization method  
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)