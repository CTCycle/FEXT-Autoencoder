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

        # First convolution layer for residual connection
        self.initial_conv = layers.Conv2D(units, kernel_size=(1,1), padding='same')

        # Dynamically create additional convolutional layers and batch normalization layers
        self.conv_layers = []
        self.batch_norm_layers = []
        
        for _ in range(num_layers - 1):  
            self.conv_layers.append(layers.Conv2D(units, kernel_size=(2,2), padding='same'))
            self.batch_norm_layers.append(layers.BatchNormalization())

    # implement forward pass through call method  
    #--------------------------------------------------------------------------    
    def call(self, inputs, training=None):
        
        residual = self.initial_conv(inputs)        
        # Pass through dynamically created convolutional and batch norm layers
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


# [SOBEL CONVOLUTIONAL BLOCKS]
###############################################################################
@keras.utils.register_keras_serializable(package='CustomLayers', name='SobelFilterConv')
class SobelFilterConv(layers.Layer):
    
    def __init__(self, **kwargs):
        super(SobelFilterConv, self).__init__(**kwargs)
        self.sobel_x = keras.ops.convert_to_tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                                    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                                    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
                                                    dtype=torch.float32)
        self.sobel_y = keras.ops.convert_to_tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                                    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                                    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], 
                                                    dtype=torch.float32)
        self.sobel_x = keras.ops.expand_dims(self.sobel_x, axis=-1)
        self.sobel_y = keras.ops.expand_dims(self.sobel_y, axis=-1)
        
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=None):
        # Apply the Sobel filter for x and y directions
        Gx = keras.ops.conv(inputs, self.sobel_x, padding="same", strides=(1,1))
        Gy = keras.ops.conv(inputs, self.sobel_y, padding="same", strides=(1,1))

        # Calculate the magnitude of the gradients
        gradients = keras.ops.sqrt(keras.ops.square(Gx) + keras.ops.square(Gy))
        
        # Optionally normalize the output
        gradients = gradients / keras.ops.amax(gradients)  
        
        return gradients
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(SobelFilterConv, self).get_config()
        config.update({})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)  

    


  
    



