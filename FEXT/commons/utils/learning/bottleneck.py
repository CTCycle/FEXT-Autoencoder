import keras
from keras import layers
import torch

from FEXT.commons.constants import CONFIG   
    
# [BOTTLENECK DECOMPRESSION AND RESHAPING]
###############################################################################
@keras.utils.register_keras_serializable(package='CustomLayers', name='CompressionLayer')
class CompressionLayer(layers.Layer):
    
    def __init__(self, units, dropout_rate=0.2, depth=4, **kwargs):
        super(CompressionLayer, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.depth = depth
        self.dense_layers = [
            layers.Dense(units, kernel_initializer='he_uniform') for _ in range(depth)]
        self.batch_norm_layers = [layers.BatchNormalization() for _ in range(depth)]        
        self.dropout = layers.Dropout(dropout_rate)

    #--------------------------------------------------------------------------
    def call(self, inputs, training=None):
        batch_size, height, width, channels = keras.ops.shape(inputs)
        sequence_dim = height * width
        reshaped = keras.ops.reshape(inputs, (batch_size, sequence_dim, channels))
        layer = reshaped
        for dense, batch_norm in zip(self.dense_layers, self.batch_norm_layers):
            layer = dense(layer)    
            layer = keras.activations.relu(layer)
            layer = batch_norm(layer, training=training)        

        return layer

    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(CompressionLayer, self).get_config()
        config.update({'units': self.units,
                       'dropout_rate': self.dropout_rate,
                       'depth': self.depth})
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
    
    def __init__(self, units, depth, **kwargs):
        super(DecompressionLayer, self).__init__(**kwargs)        
        self.units = units
        self.depth = depth
        self.dense_layers = [
            layers.Dense(units, kernel_initializer='he_uniform') for _ in range(depth)]
        self.batch_norm_layers = [layers.BatchNormalization() for _ in range(depth)]        

    #-------------------------------------------------------------------------- 
    def call(self, inputs, training=None):        
        batch_size, sequence_dims, channels = keras.ops.shape(inputs)        
        original_dims = keras.ops.sqrt(sequence_dims)
        original_dims = keras.ops.cast(original_dims, dtype=torch.int32)
        reshaped = keras.ops.reshape(
            inputs, (batch_size, original_dims, original_dims, channels))
        layer = reshaped       
        for dense, batch_norm in zip(self.dense_layers, self.batch_norm_layers):
            layer = dense(layer)    
            layer = keras.activations.relu(layer)
            layer = batch_norm(layer, training=training)   
        
        return layer

    # serialize layer for saving  
    #--------------------------------------------------------------------------    
    def get_config(self):
        config = super(DecompressionLayer, self).get_config()
        config.update({'units': self.units,
                       'depth': self.depth})
        return config

    # deserialization method  
    #--------------------------------------------------------------------------    
    @classmethod
    def from_config(cls, config):
        return cls(**config)