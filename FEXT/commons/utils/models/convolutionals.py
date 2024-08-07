import keras
from keras import activations, layers

from FEXT.commons.constants import CONFIG


# [POOLING CONVOLUTIONAL BLOCKS]
###############################################################################
@keras.utils.register_keras_serializable(package='CustomLayers', name='PooledConv')
class PooledConv(layers.Layer):
    
    def __init__(self, units, num_layers=2, **kwargs):
        super(PooledConv, self).__init__(**kwargs)
        self.units = units        
        self.num_layers = num_layers        
        self.pooling = layers.AveragePooling2D(pool_size=(2,2), padding='same')       
        self.convolutions = [layers.Conv2D(units, kernel_size=(3,3), strides=(1,1), padding='same', 
                                           activation=None) for _ in range(num_layers)]  
        self.batch_norm_layers = [layers.BatchNormalization() for _ in range(num_layers)]                 
        
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=None):
        layer = inputs
        for conv, bn in zip(self.convolutions, self.batch_norm_layers):
            layer = conv(layer)
            layer = bn(layer, training=training)
            layer = activations.relu(layer) 
        output = self.pooling(layer)           
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(PooledConv, self).get_config()
        config.update({'units': self.units,                       
                       'num_layers': self.num_layers})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)  
    

# [POOLING CONVOLUTIONAL BLOCKS]
###############################################################################
@keras.utils.register_keras_serializable(package='CustomLayers', name='TransposeConv')
class TransposeConv(layers.Layer):

    def __init__(self, units, num_layers=3, **kwargs):
        super(TransposeConv, self).__init__(**kwargs)
        self.units = units        
        self.num_layers = num_layers              
        self.upsamp = layers.UpSampling2D(size=(2,2))
        self.convolutions = [layers.Conv2DTranspose(units, kernel_size=(3,3), strides=(1,1), padding='same',  
                                                    activation=None) for _ in range(num_layers)]
        self.batch_norm_layers = [layers.BatchNormalization() for _ in range(num_layers)]             
        
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=None):
        layer = inputs
        for conv, bn in zip(self.convolutions, self.batch_norm_layers):
            layer = conv(layer) 
            layer = bn(layer, training=training)
            layer = activations.relu(layer)
        output = self.upsamp(layer)           
        
        return output  
    
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(TransposeConv, self).get_config()
        config.update({'units': self.units,                  
                       'num_layers': self.num_layers})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)     

       

    

    


  
    



