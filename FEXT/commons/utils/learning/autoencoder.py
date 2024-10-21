import keras
from keras import layers, Model
import torch

from FEXT.commons.utils.learning.convolutionals import AddNorm, PooledConv, TransposeConv, SobelFilterConv
from FEXT.commons.constants import CONFIG

       
# [AUTOENCODER MODEL]
###############################################################################
# autoencoder model built using the functional keras API. use get_model() method
# to build and compile the model (print summary as optional)
###############################################################################
class FeXTAutoEncoder: 

    def __init__(self): 
        self.img_shape = tuple(CONFIG["model"]["IMG_SHAPE"]) 
        self.apply_sobel = CONFIG["model"]["APPLY_SOBEL"]
        self.COMPILE = CONFIG["model"]["JIT_COMPILE"]
        self.jit_backend = CONFIG["model"]["JIT_BACKEND"]
        self.learning_rate = CONFIG["training"]["LEARNING_RATE"]         
        self.seed = CONFIG["SEED"]         

    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, model_summary=True):       
       
        # [ENCODER SUBMODEL]
        #----------------------------------------------------------------------
        inputs = layers.Input(shape=self.img_shape, name='image_input') 

        # perform series of convolution pooling on raw image and then concatenate
        # the results with the obtained gradients          
        layer = PooledConv(units=32, num_layers=2)(inputs)
        layer = PooledConv(units=64, num_layers=2)(layer)
        layer = PooledConv(units=128, num_layers=3)(layer)  

        if self.apply_sobel:      
            # calculate image pixels gradient using sobel filters
            # apply 2D convolution to obtained gradients
            gradients = SobelFilterConv()(inputs)
            gradients = PooledConv(units=32, num_layers=2)(gradients)
            gradients = PooledConv(units=64, num_layers=2)(gradients)
            gradients = PooledConv(units=128, num_layers=3)(gradients) 
            layer = AddNorm()([layer, gradients])

        # perform downstream convolution pooling on the concatenated vector
        # the results with the obtained gradients 
        layer = PooledConv(units=256, num_layers=3)(layer) 
        layer = PooledConv(units=512, num_layers=3)(layer)        
        layer = layers.Dropout(rate=0.1, seed=self.seed)(layer)
        encoder_output = layers.Dense(512, activation='relu',
                                      kernel_initializer='he_uniform',
                                      name='encoder_output')(layer)

        # [DECODER SUBMODEL]
        #----------------------------------------------------------------------
        layer = TransposeConv(units=512, num_layers=3)(encoder_output)       
        layer = TransposeConv(units=512, num_layers=3)(layer)
        layer = TransposeConv(units=256, num_layers=3)(layer)  
        layer = TransposeConv(units=128, num_layers=3)(layer) 
        layer = TransposeConv(units=64, num_layers=3)(layer)
        output = layers.Dense(3, activation='sigmoid', kernel_initializer='he_uniform')(layer)
       
        model = Model(inputs=inputs, outputs=output, name='FEXT_model')
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = keras.losses.MeanSquaredError()
        metric = keras.metrics.CosineSimilarity()
        model.compile(loss=loss, optimizer=opt, metrics=[metric], 
                      COMPILE=self.COMPILE)

        if self.COMPILE:
            torch.compile(model, backend=self.jit_backend, mode='default')
                
        if model_summary:
            model.summary(expand_nested=True)        

        return model
       

