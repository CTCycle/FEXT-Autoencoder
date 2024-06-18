import os
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model


# [POOLING CONVOLUTIONAL BLOCKS]
#------------------------------------------------------------------------------
@keras.utils.register_keras_serializable(package='CustomLayers', name='PooledConvBlock')
class PooledConvBlock(layers.Layer):
    def __init__(self, units, num_layers=2, seed=42, **kwargs):
        super(PooledConvBlock, self).__init__(**kwargs)
        self.units = units        
        self.num_layers = num_layers
        self.seed = seed 
        self.pooling = layers.AveragePooling2D(padding='same')       
        self.convolutions = [layers.Conv2D(units, kernel_size=(2,2), 
                                           padding='same', activation='relu') for x in range(num_layers)]         
                 
        
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=True):
        layer = inputs
        for conv in self.convolutions:
            layer = conv(layer) 
        output = self.pooling(layer)           
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(PooledConvBlock, self).get_config()
        config.update({'units': self.units,                       
                       'num_layers': self.num_layers,
                       'seed': self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)  
    

# [POOLING CONVOLUTIONAL BLOCKS]
#------------------------------------------------------------------------------
@keras.utils.register_keras_serializable(package='CustomLayers', name='TransposeConvBlock')
class TransposeConvBlock(layers.Layer):
    def __init__(self, units, num_layers=3, seed=42, **kwargs):
        super(TransposeConvBlock, self).__init__(**kwargs)
        self.units = units        
        self.num_layers = num_layers
        self.seed = seed        
        self.upsamp = layers.UpSampling2D()
        self.convolutions = [layers.Conv2DTranspose(units, 
                                                    kernel_size=(2,2), 
                                                    padding='same', 
                                                    activation='relu') for x in range(num_layers)]                
        
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=True):
        layer = inputs
        for conv in self.convolutions:
            layer = conv(layer) 
        output = self.upsamp(layer)           
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(TransposeConvBlock, self).get_config()
        config.update({'units': self.units,                  
                       'num_layers': self.num_layers,
                       'seed': self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)     

       

    

    


  
    



