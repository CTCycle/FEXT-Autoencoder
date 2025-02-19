import keras
from keras import activations, layers

from FEXT.commons.constants import CONFIG

# [CONVOLUTIONAL BLOCKS]
###############################################################################
@keras.utils.register_keras_serializable(package='CustomLayers', name='ResidualConvolutivePooling')
class ResidualConvolutivePooling(layers.Layer):
    
    def __init__(self, units, num_layers, **kwargs):
        super(ResidualConvolutivePooling, self).__init__(**kwargs)
        self.units = units
        self.num_layers = num_layers         
        self.pooling = layers.MaxPooling2D(pool_size=(2,2), padding='same')      
        self.conv_layers = [
            layers.Conv2D(units, kernel_size=(2,2), padding='same') 
            for x in range(num_layers)]                    

    # implement forward pass through call method  
    #--------------------------------------------------------------------------    
    def call(self, inputs, training=None): 
        inputs = keras.ops.mean(inputs, axis=-1, keepdims=True)
        layer = inputs        
        for conv in self.conv_layers:                       
            layer = conv(layer)
            layer = activations.relu(layer)
            layer = layers.Add()([layer, inputs])

        output = self.pooling(layer) 
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------    
    def get_config(self):
        config = super(ResidualConvolutivePooling, self).get_config()
        config.update({'units': self.units,
                      'num_layers': self.num_layers})
        return config

    # deserialization method  
    #--------------------------------------------------------------------------    
    @classmethod
    def from_config(cls, config):
        return cls(**config)    
    
    
# [TRANSPOSE CONVOLUTIONAL BLOCKS]
###############################################################################
@keras.utils.register_keras_serializable(package='CustomLayers', name='ResidualTransconvolutiveUpsampling')
class ResidualTransconvolutiveUpsampling(layers.Layer):
    
    def __init__(self, units, num_layers, **kwargs):
        super(ResidualTransconvolutiveUpsampling, self).__init__(**kwargs)
        self.units = units
        self.num_layers = num_layers              
        self.upsampling = layers.UpSampling2D(size=(2,2)) 
        self.conv_layers = [
            layers.Conv2DTranspose(units, kernel_size=(3,3), padding='same') 
            for x in range(num_layers)]
           
    # implement forward pass through call method  
    #--------------------------------------------------------------------------    
    def call(self, inputs, training=None):               
        inputs = keras.ops.mean(inputs, axis=-1, keepdims=True)   
        layer = inputs         
        for conv in self.conv_layers:           
            layer = conv(layer)   
            layer = activations.relu(layer)                     
            layer = layers.Add()([layer, inputs])
            
        output = self.upsampling(layer)  
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------    
    def get_config(self):
        config = super(ResidualTransconvolutiveUpsampling, self).get_config()
        config.update({'units': self.units,
                       'num_layers': self.num_layers})
        return config

    # deserialization method  
    #--------------------------------------------------------------------------    
    @classmethod    
    def from_config(cls, config):
        return cls(**config)


