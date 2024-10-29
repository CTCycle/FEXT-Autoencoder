import keras
from keras import layers, Model
import torch

from FEXT.commons.utils.learning.convolutionals import (StackedResidualConv, 
                                                        StackedResidualTransposeConv, 
                                                        SobelFilterConv)

       
# [AUTOENCODER MODEL]
###############################################################################
# autoencoder model built using the functional keras API. use get_model() method
# to build and compile the model (print summary as optional)
###############################################################################
class FeXTAutoEncoder: 

    def __init__(self, configuration): 
        self.img_shape = tuple(configuration["model"]["IMG_SHAPE"])
        self.use_residuals = configuration["model"]["RESIDUALS"]
        self.apply_sobel = configuration["model"]["APPLY_SOBEL"]
        self.jit_compile = configuration["model"]["JIT_COMPILE"]
        self.jit_backend = configuration["model"]["JIT_BACKEND"]
        self.learning_rate = configuration["training"]["LEARNING_RATE"]         
        self.seed = configuration["SEED"]  
        self.configuration = configuration       

    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, model_summary=True):       
       
        # [ENCODER SUBMODEL]
        #----------------------------------------------------------------------
        inputs = layers.Input(shape=self.img_shape, name='image_input') 
        
        # perform series of convolution pooling on raw image and then concatenate
        # the results with the obtained gradients          
        layer = layer = StackedResidualConv(64, residuals=self.use_residuals, num_layers=2)(inputs)     

        if self.apply_sobel:      
            # calculate image pixels gradient using sobel filters
            # apply 2D convolution to obtained gradients
            gradients = SobelFilterConv()(inputs)
            gradients = StackedResidualConv(units=64, residuals=self.use_residuals, num_layers=2)(gradients)           
            layer = layers.Add()([layer, gradients])        

        # perform downstream convolution pooling on the concatenated vector
        # the results with the obtained gradients         
        layer = layers.AveragePooling2D(pool_size=(2,2), padding='same')(layer)       
        layer = StackedResidualConv(128, residuals=self.use_residuals, num_layers=2)(layer)
        layer = layers.AveragePooling2D(pool_size=(2,2), padding='same')(layer)
        layer = StackedResidualConv(128, residuals=self.use_residuals, num_layers=2)(layer)
        layer = layers.AveragePooling2D(pool_size=(2,2), padding='same')(layer)
        layer = StackedResidualConv(units=256, residuals=self.use_residuals, num_layers=3)(layer) 
        layer = layers.AveragePooling2D(pool_size=(2,2), padding='same')(layer)
        layer = StackedResidualConv(units=256, residuals=self.use_residuals, num_layers=3)(layer)
        layer = layers.AveragePooling2D(pool_size=(2,2), padding='same')(layer) 
        layer = StackedResidualConv(units=256, residuals=self.use_residuals, num_layers=3)(layer)       
        layer = layers.SpatialDropout2D(rate=0.2, seed=self.seed)(layer)
        encoder_output = layers.Conv2D(filters=128, kernel_size=(1,1), padding='same', 
                                       activation='relu', dtype=torch.float32)(layer)
        
        # [DECODER SUBMODEL]
        #----------------------------------------------------------------------        
        layer = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(encoder_output)       
        layer = StackedResidualTransposeConv(256, residuals=self.use_residuals, num_layers=3)(layer)
        layer = layers.UpSampling2D(size=(2,2))(layer)
        layer = StackedResidualTransposeConv(256, residuals=self.use_residuals, num_layers=3)(layer)
        layer = layers.UpSampling2D(size=(2,2))(layer)
        layer = StackedResidualTransposeConv(256, residuals=self.use_residuals, num_layers=3)(layer)
        layer = layers.UpSampling2D(size=(2,2))(layer)
        layer = StackedResidualTransposeConv(128, residuals=self.use_residuals, num_layers=2)(layer)
        layer = layers.UpSampling2D(size=(2,2))(layer)
        layer = StackedResidualTransposeConv(64, residuals=self.use_residuals, num_layers=2)(layer)
        layer = layers.UpSampling2D(size=(2,2))(layer)
        layer = StackedResidualTransposeConv(64, residuals=self.use_residuals, num_layers=2)(layer)

        # Final layer to match the image shape and output channels (RGB)
        output = layers.Conv2D(filters=3, kernel_size=(1,1), padding='same', activation='sigmoid',
                               dtype=torch.float32)(layer)
        
        # define the model using the image as input and output       
        model = Model(inputs=inputs, outputs=output, name='FEXT_model')
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = keras.losses.Huber(delta=1)
        metric = [keras.metrics.CosineSimilarity()]
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=False)

        if self.jit_compile:
            model = torch.compile(model, backend=self.jit_backend, mode='default')
                
        if model_summary:
            model.summary(expand_nested=True)        

        return model
       

