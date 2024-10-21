import keras
from keras import activations, layers
import torch

from FEXT.commons.constants import CONFIG


# [CONVOLUTIONAL BLOCKS]
###############################################################################
@keras.utils.register_keras_serializable(package='CustomLayers', name='StackedResidualConv')
class StackedResidualConv(layers.Layer):
    
    def __init__(self, units, residuals=True, **kwargs):
        super(StackedResidualConv, self).__init__(**kwargs)
        self.units = units          
        self.residuals = residuals
        self.conv1 = layers.Conv2D(units, kernel_size=(1,1), padding='same') 
        self.conv2 = layers.Conv2D(units, kernel_size=(3,3), padding='same') 
        self.batch_norm1 = layers.BatchNormalization()      
        self.batch_norm2 = layers.BatchNormalization()  
               
        
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=None):
        
        layer = self.conv1(inputs)
        layer = self.batch_norm1(layer, training=training)
        layer = activations.relu(layer) 
        layer = self.conv2(layer)
        layer = self.batch_norm2(layer, training=training)    
        
        if self.residuals:
            layer = layers.Add()([layer, inputs])

        output = activations.relu(layer)
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(StackedResidualConv, self).get_config()
        config.update({'units': self.units,                      
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

    def __init__(self, units, residuals=True, **kwargs):
        super(StackedResidualTransposeConv, self).__init__(**kwargs)
        self.units = units            
        self.residuals = residuals
        self.conv1 = layers.Conv2DTranspose(units, kernel_size=(1,1),  padding='same')  
        self.conv2 = layers.Conv2DTranspose(units, kernel_size=(3,3),  padding='same')  
        self.batch_norm1 = layers.BatchNormalization()      
        self.batch_norm2 = layers.BatchNormalization()  
                                                      
             
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=None):
        layer = self.conv1(inputs)
        layer = self.batch_norm1(layer, training=training)
        layer = activations.relu(layer)
        layer = self.conv2(layer)
        layer = self.batch_norm2(layer, training=training)       
        if self.residuals:
            layer = layers.Add()([layer, inputs])

        output = activations.relu(layer)
        
        return output        
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(StackedResidualTransposeConv, self).get_config()
        config.update({'units': self.units,                       
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

    


  
    



