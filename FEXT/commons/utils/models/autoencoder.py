from tensorflow import keras
from keras import layers
from keras.models import Model

from FEXT.commons.utils.models.layers import PooledConvBlock, TransposeConvBlock
from FEXT.commons.constants import CONFIG

       
# [ENCODER MODEL]
#------------------------------------------------------------------------------
@keras.utils.register_keras_serializable(package='SubModels', name='Encoder')
class FeXTEncoder(layers.Layer):
    def __init__(self, **kwargs):
        super(FeXTEncoder, self).__init__(**kwargs)        
        self.convblock1 = PooledConvBlock(64, 2) 
        self.convblock2 = PooledConvBlock(128, 2)
        self.convblock3 = PooledConvBlock(128, 2)
        self.convblock4 = PooledConvBlock(256, 2)
        self.convblock5 = PooledConvBlock(256, 3)
        self.convblock6 = PooledConvBlock(512, 3)        
        self.dense = layers.Dense(512, activation='relu', 
                                  kernel_initializer='he_uniform')
        

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=None):        
        layer = self.convblock1(inputs, training=training) 
        layer = self.convblock2(layer, training=training)
        layer = self.convblock3(layer, training=training)
        layer = self.convblock4(layer, training=training)
        layer = self.convblock5(layer, training=training)
        layer = self.convblock6(layer, training=training)
        output = self.dense(layer)       

        return output

    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(FeXTEncoder, self).get_config()
        config.update({'seed': CONFIG["SEED"]})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config) 
    

# [DECODER MODEL]
#------------------------------------------------------------------------------
@keras.utils.register_keras_serializable(package='SubModels', name='Decoder')
class FeXTDecoder(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FeXTDecoder, self).__init__(**kwargs)
        self.convblock1 = TransposeConvBlock(512, 3)                
        self.convblock2 = TransposeConvBlock(256, 3)    
        self.convblock3 = TransposeConvBlock(256, 2)
        self.convblock4 = TransposeConvBlock(128, 2)
        self.convblock5 = TransposeConvBlock(128, 2)
        self.convblock6 = TransposeConvBlock(64, 2)
        self.dense = layers.Dense(3, activation='sigmoid', dtype='float32')

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=None):        
        
        layer = self.convblock1(inputs, training=training)
        layer = self.convblock2(layer, training=training)
        layer = self.convblock3(layer, training=training)
        layer = self.convblock4(layer, training=training)
        layer = self.convblock5(layer, training=training)
        layer = self.convblock6(layer, training=training)
        output = self.dense(layer)

        return output

    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(FeXTDecoder, self).get_config()
        config.update({'seed': CONFIG["SEED"]})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)    

    

# [AUTOENCODER MODEL]
#------------------------------------------------------------------------------
# autoencoder model built using the functional keras API. use get_model() method
# to build and compile the model (print summary as optional)
class FeXTAutoEncoder: 

    def __init__(self): 
        self.img_shape = CONFIG["model"]["IMG_SHAPE"] 
        self.learning_rate = CONFIG["training"]["LEARNING_RATE"] 
        self.xla_state = CONFIG["training"]["XLA_STATE"]  
             
        self.encoder = FeXTEncoder()
        self.decoder = FeXTDecoder()        

    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, summary=True):       
       
        inputs = layers.Input(shape=self.img_shape)           
        encoder_block = self.encoder(inputs)        
        decoder_block = self.decoder(encoder_block)        
        model = Model(inputs=inputs, outputs=decoder_block, name='FEXT_model')
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = keras.losses.MeanSquaredError()
        metric = keras.metrics.CosineSimilarity()
        model.compile(loss=loss, optimizer=opt, metrics=metric, 
                      jit_compile=self.xla_state)         
        if summary:
            model.summary(expand_nested=True)

        return model
       

