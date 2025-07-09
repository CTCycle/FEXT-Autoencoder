from keras import optimizers, losses, metrics, layers, activations, Model
from torch import compile as torch_compile

from FEXT.app.src.commons.utils.learning.training.scheduler import LinearDecayLRScheduler
from FEXT.app.src.commons.utils.learning.models.bottleneck import CompressionLayer, DecompressionLayer
from FEXT.app.src.commons.utils.learning.models.convolutionals import ResidualConvolutivePooling, ResidualTransConvolutiveUpsampling

       
# [AUTOENCODER MODEL]
###############################################################################
# autoencoder model built using the functional keras API. use get_model() method
# to build and compile the model (print summary as optional)
###############################################################################
class FeXTAutoEncoder: 

    def __init__(self, configuration : dict):  
        self.image_shape = (128, 128, 3)
        self.seed = configuration.get('training_seed', 42) 
        self.dropout_rate = configuration.get('dropout_rate', 0.2)           
        self.jit_compile = configuration.get('jit_compile', False)
        self.jit_backend = configuration.get('jit_backend', 'inductor')
        
        self.initial_neurons = configuration.get('initial_neurons', 64)  
        self.low_depth_neurons = self.initial_neurons * 2
        self.mid_depth_neurons = self.initial_neurons * 4  
        self.high_depth_neurons = self.initial_neurons * 8       

        self.configuration = configuration  
  
    #--------------------------------------------------------------------------
    def compile_model(self, model : Model, model_summary=True):
        initial_LR = self.configuration.get('initial_RL', 0.001)
        LR_schedule = initial_LR        
        if self.configuration.get('use_scheduler', False):          
            constant_lr_steps = self.configuration.get('constant_steps', 1000)   
            decay_steps = self.configuration.get('decay_steps', 500)  
            final_LR = self.configuration.get('final_LR', 0.0001)          
            LR_schedule = LinearDecayLRScheduler(
                initial_LR, constant_lr_steps, decay_steps, final_LR)  
                  
        opt = optimizers.Adam(learning_rate=LR_schedule)
        loss = losses.MeanAbsoluteError()        
        metric = [metrics.CosineSimilarity()]
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=False)                 
        # print model summary on console and run torch.compile 
        # with triton compiler and selected backend
        model.summary(expand_nested=True) if model_summary else None
        if self.jit_compile:
            model = torch_compile(model, backend=self.jit_backend, mode='default')

        return model         

    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, model_summary=True):       
       
        # [ENCODER SUBMODEL]
        #----------------------------------------------------------------------
        inputs = layers.Input(shape=self.image_shape, name='image_input')        
        
        # perform series of convolution pooling on raw image and then concatenate
        # the results with the obtained gradients          
        layer = ResidualConvolutivePooling(self.initial_neurons, num_layers=3)(inputs)              
        layer = ResidualConvolutivePooling(self.initial_neurons, num_layers=3)(layer)        
        layer = ResidualConvolutivePooling(self.low_depth_neurons, num_layers=3)(layer)        
        layer = ResidualConvolutivePooling(self.low_depth_neurons, num_layers=4)(layer)        
        layer = ResidualConvolutivePooling(self.mid_depth_neurons, num_layers=4)(layer)        
        layer = ResidualConvolutivePooling(self.mid_depth_neurons, num_layers=5)(layer)        

        # [BOTTLENECK SUBMODEL]
        #--------------------------------------------------------------------
        encoder_output = CompressionLayer(
            self.high_depth_neurons, dropout_rate=self.dropout_rate, num_layers=5)(layer) 
        decoder_input = DecompressionLayer(
            self.high_depth_neurons, num_layers=5)(encoder_output)
        
        # [DECODER SUBMODEL]
        #----------------------------------------------------------------------          
        layer = ResidualTransConvolutiveUpsampling(self.mid_depth_neurons, num_layers=5)(decoder_input)       
        layer = ResidualTransConvolutiveUpsampling(self.mid_depth_neurons, num_layers=4)(layer)       
        layer = ResidualTransConvolutiveUpsampling(self.low_depth_neurons, num_layers=4)(layer)       
        layer = ResidualTransConvolutiveUpsampling(self.low_depth_neurons, num_layers=3)(layer)       
        layer = ResidualTransConvolutiveUpsampling(self.initial_neurons, num_layers=3)(layer)       
        layer = ResidualTransConvolutiveUpsampling(self.initial_neurons, num_layers=3)(layer) 

        output = layers.Dense(3, kernel_initializer='he_uniform')(layer) 
        output = layers.BatchNormalization()(output)       
        output = activations.relu(output, max_value=1.0)  
        
        # define the model using the image as input and output       
        model = Model(inputs=inputs, outputs=output, name='FEXT_model')
        model = self.compile_model(model, model_summary=model_summary)        

        return model
       

