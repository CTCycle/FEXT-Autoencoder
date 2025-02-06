import keras
from keras import activations, layers
import torch

from FEXT.commons.constants import CONFIG


# [CONVOLUTIONAL BLOCKS]
###############################################################################
@keras.utils.register_keras_serializable(package='CustomLayers', name='StackedResidualConv')
class StackedResidualConv(layers.Layer):
    
    def __init__(self, units, num_layers, residuals=True, **kwargs):
        super(StackedResidualConv, self).__init__(**kwargs)
        self.units = units
        self.num_layers = num_layers
        self.residuals = residuals        
        self.initial_conv = layers.Conv2D(units, kernel_size=(1,1), padding='same')
       
        self.conv_layers = []
        self.batch_norm_layers = []        
        for _ in range(num_layers - 1):  
            self.conv_layers.append(layers.Conv2D(units, kernel_size=(2,2), padding='same'))
            self.batch_norm_layers.append(layers.BatchNormalization())

    # implement forward pass through call method  
    #--------------------------------------------------------------------------    
    def call(self, inputs, training=None):
        
        residual = self.initial_conv(inputs)         
        layer = residual
        for conv, batch_norm in zip(self.conv_layers, self.batch_norm_layers):
            layer = batch_norm(layer, training=training)
            layer = activations.relu(layer)
            layer = conv(layer)
        
        # Add residual connection if enabled
        if self.residuals:
            layer = layers.Add()([layer, residual])
        
        output = activations.relu(layer)
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------    
    def get_config(self):
        config = super(StackedResidualConv, self).get_config()
        config.update({'units': self.units,
                      'num_layers': self.num_layers,
                      'residuals': self.residuals})
        return config

    # deserialization method  
    #--------------------------------------------------------------------------    
    @classmethod
    def from_config(cls, config):
        return cls(**config)    
    
    
# [TRANSPOSE CONVOLUTIONAL BLOCKS]
###############################################################################
@keras.utils.register_keras_serializable(package='CustomLayers', name='StackedResidualTransposeConv')
class StackedResidualTransposeConv(layers.Layer):
    
    def __init__(self, units, num_layers, residuals=True, **kwargs):
        super(StackedResidualTransposeConv, self).__init__(**kwargs)
        self.units = units
        self.num_layers = num_layers
        self.residuals = residuals
        # First transposed convolution layer for residual connection
        self.initial_conv = layers.Conv2DTranspose(units, kernel_size=(1,1), padding='same')
        # Dynamically create additional transposed convolutional layers and batch normalization layers
        self.conv_layers = []
        self.batch_norm_layers = []        
        for _ in range(num_layers - 1):  
            self.conv_layers.append(layers.Conv2DTranspose(units, kernel_size=(3,3), padding='same'))
            self.batch_norm_layers.append(layers.BatchNormalization())

    # implement forward pass through call method  
    #--------------------------------------------------------------------------    
    def call(self, inputs, training=None):
       
        residual = self.initial_conv(inputs)        
        # Pass through dynamically created transposed convolutional and batch norm layers
        layer = residual
        for conv, batch_norm in zip(self.conv_layers, self.batch_norm_layers):
            layer = batch_norm(layer, training=training)
            layer = activations.relu(layer)
            layer = conv(layer)
        
        # Add residual connection if enabled
        if self.residuals:
            layer = layers.Add()([layer, residual])

        # Final ReLU activation
        output = activations.relu(layer)
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------    
    def get_config(self):
        config = super(StackedResidualTransposeConv, self).get_config()
        config.update({'units': self.units,
                       'num_layers': self.num_layers,
                       'residuals': self.residuals})
        return config

    # deserialization method  
    #--------------------------------------------------------------------------    
    @classmethod    
    def from_config(cls, config):
        return cls(**config)


