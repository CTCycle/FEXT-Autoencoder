import keras
from keras import activations, layers, Model
import torch

from FEXT.commons.utils.learning.metrics import WeightedMeanAbsoluteError
from FEXT.commons.utils.learning.bottleneck import CompressionLayer
from FEXT.commons.utils.learning.convolutionals import StackedResidualConv, StackedResidualTransposeConv

       
# [AUTOENCODER MODEL]
###############################################################################
# autoencoder model built using the functional keras API. use get_model() method
# to build and compile the model (print summary as optional)
###############################################################################
class FeXTAutoEncoder: 

    def __init__(self, configuration): 
        self.img_shape = tuple(configuration["model"]["IMG_SHAPE"])
        self.use_residuals = configuration["model"]["RESIDUALS"]
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
        layer = StackedResidualConv(128, residuals=self.use_residuals, num_layers=2)(inputs)         

        # perform downstream convolution pooling on the concatenated vector
        # the results with the obtained gradients         
        layer = layers.MaxPooling2D(pool_size=(2,2), padding='same')(layer)       
        layer = StackedResidualConv(128, residuals=self.use_residuals, num_layers=3)(layer)
        layer = layers.MaxPooling2D(pool_size=(2,2), padding='same')(layer)
        layer = StackedResidualConv(128, residuals=self.use_residuals, num_layers=3)(layer)
        layer = layers.MaxPooling2D(pool_size=(2,2), padding='same')(layer)
        layer = StackedResidualConv(units=256, residuals=self.use_residuals, num_layers=3)(layer) 
        layer = layers.MaxPooling2D(pool_size=(2,2), padding='same')(layer)
        layer = StackedResidualConv(units=256, residuals=self.use_residuals, num_layers=3)(layer)
        layer = layers.MaxPooling2D(pool_size=(2,2), padding='same')(layer) 
        layer = StackedResidualConv(units=512, residuals=self.use_residuals, num_layers=3)(layer)      
        layer = layers.MaxPooling2D(pool_size=(2,2), padding='same')(layer) 
        layer = StackedResidualConv(units=512, residuals=self.use_residuals, num_layers=3)(layer)        
        layer = layers.SpatialDropout2D(rate=0.2, seed=self.seed)(layer)
        layer = layers.Conv2D(filters=512, kernel_size=(1,1), padding='same', dtype=torch.float32)(layer)
        layer = layers.Conv2D(filters=512, kernel_size=(1,1), padding='same', dtype=torch.float32)(layer)
        layer = layers.Conv2D(filters=512, kernel_size=(1,1), padding='same', dtype=torch.float32)(layer)
        encoder_output = CompressionLayer(units=512)(layer) 
        
        # [DECODER SUBMODEL]
        #----------------------------------------------------------------------   
        decoder_input = DecompressionLayer(units=512)(encoder_output)            
        layer = StackedResidualTransposeConv(512, residuals=self.use_residuals, num_layers=3)(decoder_input)
        layer = layers.UpSampling2D(size=(2,2))(layer)
        layer = StackedResidualTransposeConv(512, residuals=self.use_residuals, num_layers=3)(layer)
        layer = layers.UpSampling2D(size=(2,2))(layer)
        layer = StackedResidualTransposeConv(256, residuals=self.use_residuals, num_layers=3)(layer)
        layer = layers.UpSampling2D(size=(2,2))(layer)
        layer = StackedResidualTransposeConv(256, residuals=self.use_residuals, num_layers=3)(layer)
        layer = layers.UpSampling2D(size=(2,2))(layer)
        layer = StackedResidualTransposeConv(128, residuals=self.use_residuals, num_layers=3)(layer)
        layer = layers.UpSampling2D(size=(2,2))(layer)
        layer = StackedResidualTransposeConv(128, residuals=self.use_residuals, num_layers=3)(layer)
        layer = layers.UpSampling2D(size=(2,2))(layer)
        layer = StackedResidualTransposeConv(128, residuals=self.use_residuals, num_layers=3)(layer)

        # Final layer to match the image shape and output channels (RGB)
        layer = layers.Conv2D(filters=64, kernel_size=(1,1), padding='same', dtype=torch.float32)(layer)
        layer = layers.Conv2D(filters=32, kernel_size=(1,1), padding='same', dtype=torch.float32)(layer)
        layer = layers.Conv2D(filters=3, kernel_size=(1,1), padding='same', dtype=torch.float32)(layer)
        output = activations.relu(layer, max_value=1.0)
        
        # define the model using the image as input and output       
        model = Model(inputs=inputs, outputs=output, name='FEXT_model')
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = WeightedMeanAbsoluteError(size=self.img_shape)        
        metric = [keras.metrics.CosineSimilarity()]
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=False)

        if self.jit_compile:
            model = torch.compile(model, backend=self.jit_backend, mode='default')
                
        if model_summary:
            model.summary(expand_nested=True)        

        return model
       

