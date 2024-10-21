import keras
from keras import layers, Model
import torch

from FEXT.commons.utils.learning.convolutionals import AddNorm, StackedResidualConv, StackedResidualTransposeConv, SobelFilterConv
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
        self.jit_compile = CONFIG["model"]["JIT_COMPILE"]
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
        layer = StackedResidualConv(units=64, num_layers=2, residuals=False)(inputs)
        layer = layers.AveragePooling2D(pool_size=(2,2), padding='same')(layer)
        layer = StackedResidualConv(units=64, num_layers=2, residuals=True)(layer)
        layer = layers.AveragePooling2D(pool_size=(2,2), padding='same')(layer)
        layer = StackedResidualConv(units=64, num_layers=2, residuals=True)(layer)
        layer = layers.AveragePooling2D(pool_size=(2,2), padding='same')(layer)

        if self.apply_sobel:      
            # calculate image pixels gradient using sobel filters
            # apply 2D convolution to obtained gradients
            gradients = SobelFilterConv()(inputs)
            gradients = StackedResidualConv(units=64, num_layers=2, residuals=False)(gradients)
            gradients = layers.AveragePooling2D(pool_size=(2,2), padding='same')(gradients)
            gradients = StackedResidualConv(units=64, num_layers=2, residuals=True)(gradients)
            gradients = layers.AveragePooling2D(pool_size=(2,2), padding='same')(gradients)
            gradients = StackedResidualConv(units=64, num_layers=2, residuals=True)(gradients)
            gradients = layers.AveragePooling2D(pool_size=(2,2), padding='same')(gradients)
            layer = AddNorm()([layer, gradients])

        # perform downstream convolution pooling on the concatenated vector
        # the results with the obtained gradients 
        layer = StackedResidualConv(units=64, num_layers=2, residuals=True)(layer) 
        layer = layers.AveragePooling2D(pool_size=(2,2), padding='same')(layer)
        layer = StackedResidualConv(units=128, num_layers=3, residuals=False)(layer)
        layer = layers.AveragePooling2D(pool_size=(2,2), padding='same')(layer) 
        layer = StackedResidualConv(units=256, num_layers=3, residuals=False)(layer)       
        layer = layers.Dropout(rate=0.2, seed=self.seed)(layer)
        encoder_output = layers.Dense(256, activation='relu',
                                      kernel_initializer='he_uniform',
                                      name='encoder_output')(layer)
        
        # [DECODER SUBMODEL]
        #----------------------------------------------------------------------        
        layer = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(encoder_output)  # Match encoder's 512 output to decoder's input
        layer = layers.Dropout(rate=0.2, seed=self.seed)(layer)

        layer = StackedResidualTransposeConv(units=256, num_layers=3, residuals=True)(layer)
        layer = layers.UpSampling2D(size=(2,2))(layer)
        layer = StackedResidualTransposeConv(units=128, num_layers=3, residuals=False)(layer)
        layer = layers.UpSampling2D(size=(2,2))(layer)
        layer = StackedResidualTransposeConv(units=64, num_layers=2, residuals=False)(layer)
        layer = layers.UpSampling2D(size=(2,2))(layer)
        layer = StackedResidualTransposeConv(units=64, num_layers=2, residuals=True)(layer)
        layer = layers.UpSampling2D(size=(2,2))(layer)
        layer = StackedResidualTransposeConv(units=64, num_layers=2, residuals=True)(layer)
        layer = layers.UpSampling2D(size=(2,2))(layer)

        # Final layer to match the image shape and output channels (RGB)
        output = layers.Conv2D(filters=3, kernel_size=(3,3), activation='sigmoid', 
                               padding='same', name='decoder_output')(layer)
        
        model = Model(inputs=inputs, outputs=output, name='FEXT_model')
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = keras.losses.MeanSquaredError()
        metric = keras.metrics.CosineSimilarity()
        model.compile(loss=loss, optimizer=opt, metrics=[metric], jit_compile=False)

        if self.jit_compile:
            torch.compile(model, backend=self.jit_backend, mode='default')
                
        if model_summary:
            model.summary(expand_nested=True)        

        return model
       

