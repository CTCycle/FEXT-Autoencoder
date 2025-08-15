from keras import optimizers, losses, metrics, layers, activations, Model
from torch import compile as torch_compile

from FEXT.app.utils.learning.training.scheduler import LinearDecayLRScheduler
from FEXT.app.utils.learning.models.bottleneck import CompressionLayer, DecompressionLayer
from FEXT.app.utils.learning.models.convolutionals import ResidualConvolutivePooling, ResidualTransConvolutiveUpsampling

       
# [AUTOENCODER MODEL]
###############################################################################
# autoencoder model built using the functional keras API. use get_model() method
# to build and compile the model (print summary as optional)
###############################################################################
class FeXTAutoEncoders: 

    def __init__(self, configuration : dict):
        self.image_height = configuration.get('image_height', 256) 
        self.image_width = configuration.get('image_width', 256)   
        self.channels = 1 if configuration.get('use_grayscale', False) else 3
        self.image_shape = (self.image_height, self.image_width, self.channels)         
        self.selected_model = configuration.get('selected_model', None)
        self.dropout_rate = configuration.get('dropout_rate', 0.2)           
        self.jit_compile = configuration.get('jit_compile', False)
        self.jit_backend = configuration.get('jit_backend', 'inductor')
        self.seed = configuration.get('training_seed', 42)
        self.configuration = configuration  

        self.models = {'FeXTAERedux' : self.build_redux_autoencoder,
                       'FeXTAEMedium' : self.build_medium_autoencoder,
                       'FeXTAELarge' : self.build_large_autoencoder}
  
    #--------------------------------------------------------------------------
    def compile_model(self, model : Model, model_summary=True):
        initial_LR = self.configuration.get('initial_RL', 0.001)
        LR_schedule = initial_LR        
        if self.configuration.get('use_scheduler', False):          
            constant_LR_steps = self.configuration.get('constant_steps', 1000)   
            decay_steps = self.configuration.get('decay_steps', 500)  
            target_LR = self.configuration.get('target_LR', 0.0001)          
            LR_schedule = LinearDecayLRScheduler(
                initial_LR, constant_LR_steps, decay_steps, target_LR)  
                  
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
    def get_selected_model(self, model_summary=True):
        model = self.models[self.selected_model](model_summary)
        model = self.compile_model(model, model_summary=model_summary) 

        return model          

    # build model given the architecture
    #--------------------------------------------------------------------------
    def build_redux_autoencoder(self):         
        inputs = layers.Input(shape=self.image_shape, name='image_input')  
        # perform series of convolution pooling on raw image and then concatenate
        # the results with the obtained gradients          
        layer = ResidualConvolutivePooling(16, num_layers=2)(inputs)              
        layer = ResidualConvolutivePooling(32, num_layers=2)(layer)        
        layer = ResidualConvolutivePooling(64, num_layers=2)(layer)        
        layer = ResidualConvolutivePooling(128, num_layers=3)(layer)        
        layer = ResidualConvolutivePooling(128, num_layers=3)(layer)
        # bottleneck
        encoder_output = CompressionLayer(
            256, dropout_rate=self.dropout_rate, num_layers=3)(layer) 
        decoder_input = DecompressionLayer(
            256, num_layers=3)(encoder_output)
        
        # decoder         
        layer = ResidualTransConvolutiveUpsampling(128, num_layers=3)(decoder_input)       
        layer = ResidualTransConvolutiveUpsampling(128, num_layers=3)(layer)       
        layer = ResidualTransConvolutiveUpsampling(64, num_layers=2)(layer)       
        layer = ResidualTransConvolutiveUpsampling(32, num_layers=2)(layer)  
        layer = ResidualTransConvolutiveUpsampling(16, num_layers=2)(layer) 
        # image reconstruction head
        output = layers.Dense(self.channels, kernel_initializer='he_uniform')(layer) 
        output = layers.BatchNormalization()(output)       
        output = activations.relu(output, max_value=1.0)
        # define the model using the image as input and output       
        model = Model(inputs=inputs, outputs=output, name='FEXTReduxAE')

        return model
    
    # build model given the architecture
    #--------------------------------------------------------------------------
    def build_medium_autoencoder(self):         
        inputs = layers.Input(shape=self.image_shape, name='image_input')  
        # perform series of convolution pooling on raw image and then concatenate
        # the results with the obtained gradients          
        layer = ResidualConvolutivePooling(32, num_layers=3)(inputs)   
        layer = ResidualConvolutivePooling(64, num_layers=3)(layer)            
        layer = ResidualConvolutivePooling(128, num_layers=3)(layer)        
        layer = ResidualConvolutivePooling(128, num_layers=3)(layer)        
        layer = ResidualConvolutivePooling(256, num_layers=4)(layer)        
        layer = ResidualConvolutivePooling(256, num_layers=4)(layer)
        # bottleneck
        encoder_output = CompressionLayer(
            512, dropout_rate=self.dropout_rate, num_layers=4)(layer) 
        decoder_input = DecompressionLayer(
            512, num_layers=4)(encoder_output)
        
        # decoder         
        layer = ResidualTransConvolutiveUpsampling(256, num_layers=4)(decoder_input)       
        layer = ResidualTransConvolutiveUpsampling(256, num_layers=4)(layer)       
        layer = ResidualTransConvolutiveUpsampling(128, num_layers=3)(layer)    
        layer = ResidualTransConvolutiveUpsampling(128, num_layers=3)(layer)       
        layer = ResidualTransConvolutiveUpsampling(64, num_layers=3)(layer)  
        layer = ResidualTransConvolutiveUpsampling(32, num_layers=3)(layer) 
        # image reconstruction head
        output = layers.Dense(self.channels, kernel_initializer='he_uniform')(layer) 
        output = layers.BatchNormalization()(output)       
        output = activations.relu(output, max_value=1.0)
        # define the model using the image as input and output       
        model = Model(inputs=inputs, outputs=output, name='FEXTMediumAE')

        return model
       
    # build model given the architecture
    #--------------------------------------------------------------------------
    def build_large_autoencoder(self):         
        inputs = layers.Input(shape=self.image_shape, name='image_input')  
        # perform series of convolution pooling on raw image and then concatenate
        # the results with the obtained gradients          
        layer = ResidualConvolutivePooling(32, num_layers=3)(inputs)   
        layer = ResidualConvolutivePooling(64, num_layers=3)(layer)            
        layer = ResidualConvolutivePooling(128, num_layers=3)(layer)        
        layer = ResidualConvolutivePooling(128, num_layers=3)(layer)        
        layer = ResidualConvolutivePooling(256, num_layers=4)(layer)        
        layer = ResidualConvolutivePooling(256, num_layers=4)(layer)
        # bottleneck
        encoder_output = CompressionLayer(
            512, dropout_rate=self.dropout_rate, num_layers=4)(layer) 
        decoder_input = DecompressionLayer(
            512, num_layers=4)(encoder_output)
        
        # decoder         
        layer = ResidualTransConvolutiveUpsampling(256, num_layers=4)(decoder_input)       
        layer = ResidualTransConvolutiveUpsampling(256, num_layers=4)(layer)       
        layer = ResidualTransConvolutiveUpsampling(128, num_layers=3)(layer)    
        layer = ResidualTransConvolutiveUpsampling(128, num_layers=3)(layer)       
        layer = ResidualTransConvolutiveUpsampling(64, num_layers=3)(layer)  
        layer = ResidualTransConvolutiveUpsampling(32, num_layers=3)(layer) 
        # image reconstruction head
        output = layers.Dense(self.channels, kernel_initializer='he_uniform')(layer) 
        output = layers.BatchNormalization()(output)       
        output = activations.relu(output, max_value=1.0)
        # define the model using the image as input and output       
        model = Model(inputs=inputs, outputs=output, name='FEXTMediumAE')

        return model