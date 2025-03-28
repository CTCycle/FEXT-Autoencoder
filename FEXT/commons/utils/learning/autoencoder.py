import keras
from keras import layers, activations, metrics, losses, Model
import torch

from FEXT.commons.utils.learning.scheduler import LinearDecayLRScheduler
from FEXT.commons.utils.learning.bottleneck import CompressionLayer, DecompressionLayer
from FEXT.commons.utils.learning.convolutionals import ResidualConvolutivePooling, ResidualTransConvolutiveUpsampling

       
# [AUTOENCODER MODEL]
###############################################################################
# autoencoder model built using the functional keras API. use get_model() method
# to build and compile the model (print summary as optional)
###############################################################################
class FeXTAutoEncoder: 

    def __init__(self, configuration):  
        self.image_shape = (128, 128, 3)
        self.seed = configuration["SEED"]       
        self.jit_compile = configuration["model"]["JIT_COMPILE"]
        self.jit_backend = configuration["model"]["JIT_BACKEND"]
        self.scheduler_config = configuration["training"]["LR_SCHEDULER"]
        self.initial_lr = self.scheduler_config["INITIAL_LR"]
        self.constant_lr_steps = self.scheduler_config["CONSTANT_STEPS"]       
        self.decay_steps = self.scheduler_config["DECAY_STEPS"]  
        self.final_lr = self.scheduler_config["FINAL_LR"]         
        self.configuration = configuration       

    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, model_summary=True):       
       
        # [ENCODER SUBMODEL]
        #----------------------------------------------------------------------
        inputs = layers.Input(shape=self.image_shape, name='image_input')        
        
        # perform series of convolution pooling on raw image and then concatenate
        # the results with the obtained gradients          
        layer = ResidualConvolutivePooling(64, num_layers=4)(inputs)              
        layer = ResidualConvolutivePooling(64, num_layers=4)(layer)        
        layer = ResidualConvolutivePooling(128, num_layers=5)(layer)        
        layer = ResidualConvolutivePooling(units=128, num_layers=5)(layer)        
        layer = ResidualConvolutivePooling(units=256, num_layers=6)(layer)        
        layer = ResidualConvolutivePooling(units=512, num_layers=6)(layer)        

        # [BOTTLENECK SUBMODEL]
        #--------------------------------------------------------------------
        encoder_output = CompressionLayer(units=512, dropout_rate=0.2, num_layers=5)(layer) 
        decoder_input = DecompressionLayer(units=512, num_layers=5)(encoder_output)
        
        # [DECODER SUBMODEL]
        #----------------------------------------------------------------------          
        layer = ResidualTransConvolutiveUpsampling(512, num_layers=6)(decoder_input)       
        layer = ResidualTransConvolutiveUpsampling(256, num_layers=6)(layer)       
        layer = ResidualTransConvolutiveUpsampling(128, num_layers=5)(layer)       
        layer = ResidualTransConvolutiveUpsampling(128, num_layers=5)(layer)       
        layer = ResidualTransConvolutiveUpsampling(64, num_layers=4)(layer)       
        layer = ResidualTransConvolutiveUpsampling(64, num_layers=4)(layer) 

        output = layers.Dense(3, kernel_initializer='he_uniform')(layer) 
        output = layers.BatchNormalization()(output)       
        output = activations.relu(output, max_value=1.0)   
        
        # define the model using the image as input and output       
        model = Model(inputs=inputs, outputs=output, name='FEXT_model')
        lr_schedule = LinearDecayLRScheduler(
            self.initial_lr, self.constant_lr_steps, self.decay_steps, self.final_lr)            
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)
        loss = losses.MeanAbsoluteError()        
        metric = [metrics.CosineSimilarity()]
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=False)        
                
        if model_summary:
            model.summary(expand_nested=True)   

        if self.jit_compile:
            model = torch.compile(model, backend=self.jit_backend, mode='default')    

        return model
       

