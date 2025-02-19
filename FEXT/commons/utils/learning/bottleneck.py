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
        self.dense1 = layers.Dense(units, kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(units, kernel_initializer='he_uniform')
        self.batch_norm1 = layers.BatchNormalization()
        self.batch_norm2 = layers.BatchNormalization()
        self.dropout = layers.Dropout(dropout_rate)

    #--------------------------------------------------------------------------
    def call(self, inputs, training=None):
        batch_size, height, width, channels = keras.ops.shape(inputs)
        sequence_dim = height * width
        reshaped = keras.ops.reshape(inputs, (batch_size, sequence_dim, channels))
        layer = self.dense1(reshaped)
        layer = keras.activations.relu(layer)
        layer = self.batch_norm1(layer, training=training)        
        layer = self.dropout(layer, training=training)
        layer = self.dense2(layer)        
        layer = keras.activations.relu(layer)
        output = self.batch_norm2(layer, training=training)
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
    
    
# [BOTTLENECK DECOMPRESSION AND RESHAPING]
###############################################################################
@keras.utils.register_keras_serializable(package='CustomLayers', name='DecompressionLayer')
class DecompressionLayer(layers.Layer):
    
    def __init__(self, units, **kwargs):
        super(DecompressionLayer, self).__init__(**kwargs)        
        self.units = units
        self.dense1 = layers.Dense(units, kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(units, kernel_initializer='he_uniform')
        self.batch_norm1 = layers.BatchNormalization()
        self.batch_norm2 = layers.BatchNormalization()

    #-------------------------------------------------------------------------- 
    def call(self, inputs, training=None):
        
        batch_size, sequence_dim, channels = keras.ops.shape(inputs)        
        original_dim = keras.ops.sqrt(sequence_dim)
        original_dim = keras.ops.cast(original_dim, torch.int32)        
        layer = self.dense1(inputs)
        layer = self.batch_norm1(layer, training=training)
        layer = keras.activations.relu(layer)        
        layer = self.dense2(layer)
        layer = self.batch_norm2(layer, training=training)      
        layer = keras.activations.relu(layer)
        output = keras.ops.reshape(layer, (batch_size, original_dim, original_dim, channels))
        
        return output

    # serialize layer for saving  
    #--------------------------------------------------------------------------    
    def get_config(self):
        config = super(DecompressionLayer, self).get_config()
        config.update({'units': self.units})
        return config

    # deserialization method  
    #--------------------------------------------------------------------------    
    @classmethod
    def from_config(cls, config):
        return cls(**config)