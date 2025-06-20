import keras

    
# [BOTTLENECK DECOMPRESSION AND RESHAPING]
###############################################################################
@keras.saving.register_keras_serializable(package='CustomLayers', name='CompressionLayer')
class CompressionLayer(keras.layers.Layer):
    
    def __init__(self, units, dropout_rate=0.2, num_layers=4, seed=42, **kwargs):
        super(CompressionLayer, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.seed = seed
        self.dense_layers = [
            keras.layers.Dense(units, kernel_initializer='he_uniform') for _ in range(num_layers)]
        self.batch_norm_layers = [keras.layers.BatchNormalization() for _ in range(num_layers)]        
        self.dropout = keras.layers.Dropout(dropout_rate, seed=self.seed)

    #--------------------------------------------------------------------------
    def call(self, inputs, training=None):
        batch_size, height, width, channels = keras.ops.shape(inputs)
        sequence_dim = height * width
        reshaped = keras.ops.reshape(
            inputs, (batch_size, sequence_dim, channels))
        layer = self.dropout(reshaped, training=training)  
        for dense, batch_norm in zip(self.dense_layers, self.batch_norm_layers):
            layer = dense(layer)
            layer = batch_norm(layer, training=training)     
            layer = keras.activations.relu(layer)                   

        return layer

    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(CompressionLayer, self).get_config()
        config.update({'units': self.units,
                       'dropout_rate': self.dropout_rate,
                       'num_layers': self.num_layers,
                       'seed': self.seed})
        return config

    # deserialization method  
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
# [BOTTLENECK DECOMPRESSION AND RESHAPING]
###############################################################################
@keras.saving.register_keras_serializable(package='CustomLayers', name='DecompressionLayer')
class DecompressionLayer(keras.layers.Layer):
    
    def __init__(self, units=256, num_layers=3, seed=42, **kwargs):
        super(DecompressionLayer, self).__init__(**kwargs)        
        self.units = units
        self.num_layers = num_layers
        self.seed = seed
        self.dense_layers = [
            keras.layers.Dense(units, kernel_initializer='he_uniform') 
            for _ in range(num_layers)]
        self.batch_norm_layers = [
            keras.layers.BatchNormalization() for _ in range(num_layers)]        

    #-------------------------------------------------------------------------- 
    def call(self, inputs, training=None):        
        batch_size, sequence_dims, channels = keras.ops.shape(inputs)        
        original_dims = keras.ops.sqrt(sequence_dims)
        original_dims = keras.ops.cast(original_dims, dtype='int32')        
        layer = keras.ops.reshape(
            inputs, (batch_size, original_dims, original_dims, channels))     
        for dense, batch_norm in zip(self.dense_layers, self.batch_norm_layers):
            layer = dense(layer) 
            layer = batch_norm(layer, training=training)    
            layer = keras.activations.relu(layer)              
        
        return layer

    # serialize layer for saving  
    #--------------------------------------------------------------------------    
    def get_config(self):
        config = super(DecompressionLayer, self).get_config()
        config.update({'units': self.units,
                       'num_layers': self.num_layers,
                       'seed': self.seed})
        return config

    # deserialization method  
    #--------------------------------------------------------------------------    
    @classmethod
    def from_config(cls, config):
        return cls(**config)