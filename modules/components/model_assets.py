import os
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from keras.models import Model

    
# [CALLBACK FOR REAL TIME TRAINING MONITORING]
#==============================================================================
# Callback to check real time model history and visualize it through custom plot
#==============================================================================
class RealTimeHistory(keras.callbacks.Callback):    
     
    def __init__(self, plot_path, validation=True):        
        super().__init__()
        self.plot_path = plot_path
        self.epochs = []
        self.loss_hist = []
        self.metric_hist = []
        self.loss_val_hist = []        
        self.metric_val_hist = []
        self.validation = validation 
    
    def on_epoch_end(self, epoch, logs = {}):        
        if epoch % 2 == 0:                    
            self.epochs.append(epoch)
            self.loss_hist.append(logs[list(logs.keys())[0]])
            self.metric_hist.append(logs[list(logs.keys())[1]])
            if self.validation==True:
                self.loss_val_hist.append(logs[list(logs.keys())[2]])            
                self.metric_val_hist.append(logs[list(logs.keys())[3]])
        if epoch % 2 == 0:           
            fig_path = os.path.join(self.plot_path, 'training_history.jpeg')
            plt.subplot(2, 1, 1)
            plt.plot(self.epochs, self.loss_hist, label = 'training loss')
            if self.validation==True:
                plt.plot(self.epochs, self.loss_val_hist, label = 'validation loss')
                plt.legend(loc = 'best', fontsize = 8)
            plt.title('Loss plot')
            plt.ylabel('Mean Square Error')
            plt.xlabel('epoch')
            plt.subplot(2, 1, 2)
            plt.plot(self.epochs, self.metric_hist, label='train metrics') 
            if self.validation==True: 
                plt.plot(self.epochs, self.metric_val_hist, label='validation metrics') 
                plt.legend(loc = 'best', fontsize = 8)
            plt.title('metrics plot')
            plt.ylabel('Cosine Similarity')
            plt.xlabel('epoch')       
            plt.tight_layout()
            plt.savefig(fig_path, bbox_inches='tight', format='jpeg', dpi=300)            
            plt.close() 
            
            
# [CUSTOM DATA GENERATOR FOR TRAINING]
#==============================================================================
# Generate batches of inputs and outputs using a custom generator function and 
# tf.dataset with prefetching
#==============================================================================
class DataGenerator(keras.utils.Sequence):

    def __init__(self, dataframe, batch_size, picture_shape=(244, 244, 3), shuffle=True,
                  augmentation=True, normalization=True):        
        self.dataframe = dataframe
        self.path_col = 'images path'       
        self.num_of_samples = dataframe.shape[0]
        self.picture_shape = picture_shape
        self.batch_size = batch_size  
        self.batch_index = 0 
        self.augmentation = augmentation
        self.normalization = normalization             
        self.shuffle = shuffle
        self.on_epoch_end()       

    # define length of the custom generator      
    #--------------------------------------------------------------------------
    def __len__(self):
        length = int(np.floor(self.num_of_samples)/self.batch_size)
        return length
    
    # define method to get X and Y data through custom functions, and subsequently
    # create a batch of data converted to tensors
    #--------------------------------------------------------------------------
    def __getitem__(self, idx): 
        path_batch = self.dataframe[self.path_col][idx * self.batch_size:(idx + 1) * self.batch_size]           
        x1_batch = [self.__images_generation(image_path) for image_path in path_batch]        
        X1_tensor = tf.convert_to_tensor(x1_batch)
        Y_tensor = X1_tensor  
        return X1_tensor, Y_tensor
    
    # define method to perform data operations on epoch end
    #--------------------------------------------------------------------------
    def on_epoch_end(self):        
        self.indexes = np.arange(self.num_of_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # define method perform data augmentation    
    #--------------------------------------------------------------------------
    def __images_augmentation(self, image):
        pp_image = tf.keras.preprocessing.image.random_shift(image, 0.2, 0.3)
        pp_image = tf.image.random_flip_left_right(pp_image)
        pp_image = tf.image.random_flip_up_down(pp_image)
        return pp_image        

    # define method to load images 
    #--------------------------------------------------------------------------
    def __images_generation(self, path):
        image = tf.io.read_file(path)
        rgb_image = tf.image.decode_image(image, channels=3)
        rgb_image = tf.image.resize(rgb_image, self.picture_shape[:-1])        
        if self.augmentation==True:
            rgb_image = self.__images_augmentation(rgb_image)
        if self.normalization==True:
            rgb_image = rgb_image/255.0
        return rgb_image    
    
    # define method to call the next elements of the generator    
    #--------------------------------------------------------------------------
    def next(self):
        next_index = (self.batch_index + 1) % self.__len__()
        self.batch_index = next_index
        return self.__getitem__(next_index)
    
    
# [POOLING CONVOLUTIONAL BLOCKS]
#==============================================================================
# Positional embedding custom layer
#==============================================================================
@keras.utils.register_keras_serializable(package='CustomLayers', name='PooledConvBlock')
class PooledConvBlock(layers.Layer):
    def __init__(self, units, kernel_size, num_layers=2, seed=42, **kwargs):
        super(PooledConvBlock, self).__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.seed = seed        
        self.convolutions = [layers.Conv2D(units, kernel_size=kernel_size, padding='same', activation='relu') for x in range(num_layers)]         
        self.pooling = layers.AveragePooling2D(padding='same')         
        
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=True):
        layer = inputs
        for conv in self.convolutions:
            layer = conv(layer) 
        output = self.pooling(layer)           
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(PooledConvBlock, self).get_config()
        config.update({'units': self.units,
                       'kernel_size': self.kernel_size,
                       'num_layers': self.num_layers,
                       'seed': self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)  
    

# [POOLING CONVOLUTIONAL BLOCKS]
#==============================================================================
# Positional embedding custom layer
#==============================================================================
@keras.utils.register_keras_serializable(package='CustomLayers', name='TransposeConvBlock')
class TransposeConvBlock(layers.Layer):
    def __init__(self, units, kernel_size, num_layers=3, seed=42, **kwargs):
        super(TransposeConvBlock, self).__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.seed = seed        
        self.upsamp = layers.UpSampling2D()
        self.convolutions = [layers.Conv2DTranspose(units, 
                                                    kernel_size=kernel_size, 
                                                    padding='same', 
                                                    activation='relu') for x in range(num_layers)]                
        
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=True):
        layer = inputs
        for conv in self.convolutions:
            layer = conv(layer) 
        output = self.upsamp(layer)           
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(TransposeConvBlock, self).get_config()
        config.update({'units': self.units,
                       'kernel_size': self.kernel_size,
                       'num_layers': self.num_layers,
                       'seed': self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)     

       
# [MACHINE LEARNING MODELS]
#==============================================================================
# collection of model and submodels
#==============================================================================
@keras.utils.register_keras_serializable(package='SubModels', name='Encoder')
class FeXTEncoder(layers.Layer):
    def __init__(self, kernel_size, picture_shape=(144, 144, 3), seed=42, **kwargs):
        super(FeXTEncoder, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.seed = seed        
        self.picture_shape = picture_shape
        self.convblock1 = PooledConvBlock(64, kernel_size, 2, seed)
        self.convblock2 = PooledConvBlock(128, kernel_size, 2, seed)
        self.convblock3 = PooledConvBlock(256, kernel_size, 3, seed)
        self.convblock4 = PooledConvBlock(256, kernel_size, 3, seed)
        self.convblock5 = PooledConvBlock(512, kernel_size, 3, seed)
        self.pooling = layers.MaxPooling2D(pool_size=(4, 4), strides=4)              
        self.dense2 = layers.Dense(2048, activation='LeakyReLU', kernel_initializer='he_uniform')
        self.flatten = layers.Flatten()        

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=True):
        layer = inputs
        layer = self.convblock1(layer)
        layer = self.convblock2(layer)
        layer = self.convblock3(layer)
        layer = self.convblock4(layer)
        layer = self.convblock5(layer)
        layer = self.pooling(layer)
        layer = self.flatten(layer)        
        output = self.dense2(layer)

        return output

    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(FeXTEncoder, self).get_config()
        config.update({'kernel_size': self.kernel_size,                                             
                       'picture_shape': self.picture_shape,
                       'seed': self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config) 

# [MACHINE LEARNING MODELS]
#==============================================================================
# collection of model and submodels
#==============================================================================
@keras.utils.register_keras_serializable(package='SubModels', name='Decoder')
class FeXTDecoder(keras.layers.Layer):
    def __init__(self, kernel_size, seed=42, **kwargs):
        super(FeXTDecoder, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.seed = seed  
        self.dense = layers.Dense(2048, activation='LeakyReLU', kernel_initializer='he_uniform')       
        self.reshape = layers.Reshape((2, 2, 512)) 
        self.upsamp = layers.UpSampling2D(size=(4, 4), input_shape=(2, 2, 512))
        self.convblock1 = TransposeConvBlock(512, kernel_size, 3, seed)    
        self.convblock2 = TransposeConvBlock(512, kernel_size, 3, seed)
        self.convblock3 = TransposeConvBlock(256, kernel_size, 3, seed)
        self.convblock4 = TransposeConvBlock(128, kernel_size, 2, seed)
        self.convblock5 = TransposeConvBlock(64, kernel_size, 2, seed)
        self.dense = layers.Dense(3, activation='sigmoid', dtype='float32')

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=True):
        
        layer = self.reshape(inputs)
        layer = self.upsamp(layer)
        layer = self.convblock1(layer)
        layer = self.convblock2(layer)
        layer = self.convblock3(layer)
        layer = self.convblock4(layer)
        layer = self.convblock5(layer)
        output = self.dense(layer)

        return output

    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(FeXTDecoder, self).get_config()
        config.update({'kernel_size': self.kernel_size,
                       'seed': self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)       
    
# [LEARNING RATE SCHEDULER]
#==============================================================================
# Use TensorFlow's conditional to handle the tensor-based condition, such as
# building an autograph for training   
#==============================================================================
@keras.utils.register_keras_serializable(package='LRScheduler')
class LRScheduler(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, decay_steps, decay_rate, warmup_steps=0):
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        self.warmup_lr = initial_lr * warmup_steps
        
    # call on step
    #--------------------------------------------------------------------------
    def __call__(self, step):               
        step = step + 1
        step_tensor = tf.convert_to_tensor(step, dtype=tf.float32)
        if self.warmup_steps > 0:
            warmup_lr = self.warmup_lr * (step_tensor/self.warmup_steps)
        else:
            warmup_lr = self.initial_lr

        decay_lr = self.initial_lr * (self.decay_rate ** ((step - self.warmup_steps) // self.decay_steps))
        lr = tf.cond(tf.math.less(step_tensor, self.warmup_steps),
                     lambda: warmup_lr,
                     lambda: decay_lr)
        
        return lr
    
    # custom configurations
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(LRScheduler, self).get_config()
        config.update({'initial_lr': self.initial_lr,
                       'decay_steps': self.decay_steps,
                       'decay_rate': self.decay_rate,
                       'warmup_steps': self.warmup_steps})
        return config        
    
    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

# [MACHINE LEARNING MODELS]
#==============================================================================
# autoencoder model built using the functional keras API. use get_model() method
# to build and compile the model (print summary as optional)
#==============================================================================
class FeXTAutoEncoder: 

    def __init__(self, learning_rate, kernel_size, picture_shape=(144, 144, 3), 
                 seed=42, XLA_state=False):         
        self.learning_rate = learning_rate
        self.kernel_size = kernel_size
        self.seed = seed        
        self.picture_shape = picture_shape         
        self.XLA_state = XLA_state
        self.encoder = FeXTEncoder(kernel_size, picture_shape, seed)
        self.decoder = FeXTDecoder(kernel_size, seed)        

    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, summary=True):       
       
        inputs = layers.Input(shape = self.picture_shape)           
        encoder_block = self.encoder(inputs)        
        decoder_block = self.decoder(encoder_block)        
        model = Model(inputs=inputs, outputs=decoder_block, name='FEXT_model')
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = keras.losses.MeanSquaredError()
        metric = keras.metrics.CosineSimilarity()
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=self.XLA_state)         
        if summary==True:
            model.summary(expand_nested=True)

        return model
       

# [TOOLS FOR TRAINING MACHINE LEARNING MODELS]
#==============================================================================
# Collection of methods for machine learning training and tensorflow settings
#==============================================================================
class ModelTraining:    
       
    def __init__(self, device='default', seed=42, use_mixed_precision=False):                            
        np.random.seed(seed)
        tf.random.set_seed(seed)         
        self.available_devices = tf.config.list_physical_devices()
        print('-------------------------------------------------------------------------------')        
        print('The current devices are available: ')
        print('-------------------------------------------------------------------------------')
        for dev in self.available_devices:
            print()
            print(dev)
        print()
        print('-------------------------------------------------------------------------------')
        if device == 'GPU':
            self.physical_devices = tf.config.list_physical_devices('GPU')
            if not self.physical_devices:
                print('No GPU found. Falling back to CPU')
                tf.config.set_visible_devices([], 'GPU')
            else:
                if use_mixed_precision == True:
                    policy = keras.mixed_precision.Policy('mixed_float16')
                    keras.mixed_precision.set_global_policy(policy) 
                tf.config.set_visible_devices(self.physical_devices[0], 'GPU')
                os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'                 
                print('GPU is set as active device')
            print('-------------------------------------------------------------------------------')
            print()        
        elif device == 'CPU':
            tf.config.set_visible_devices([], 'GPU')
            print('CPU is set as active device')
            print('-------------------------------------------------------------------------------')
            print()   
    
    #-------------------------------------------------------------------------- 
    def model_parameters(self, parameters_dict, savepath):

        '''
        Saves the model parameters to a JSON file. The parameters are provided 
        as a dictionary and are written to a file named 'model_parameters.json' 
        in the specified directory.

        Keyword arguments:
            parameters_dict (dict): A dictionary containing the parameters to be saved.
            savepath (str): The directory path where the parameters will be saved.

        Returns:
            None       

        '''
        path = os.path.join(savepath, 'model_parameters.json')      
        with open(path, 'w') as f:
            json.dump(parameters_dict, f)

        
# [INFERENCE]
#==============================================================================
# Collection of methods for machine learning validation and model evaluation
#==============================================================================
class Inference:

    #-------------------------------------------------------------------------- 
    def load_pretrained_model(self, path):

        '''
        Load pretrained keras model (in folders) from the specified directory. 
        If multiple model directories are found, the user is prompted to select one,
        while if only one model directory is found, that model is loaded directly.
        If `load_parameters` is True, the function also loads the model parameters 
        from the target .json file in the same directory. 

        Keyword arguments:
            path (str): The directory path where the pretrained models are stored.
            load_parameters (bool, optional): If True, the function also loads the 
                                              model parameters from a JSON file. 
                                              Default is True.

        Returns:
            model (keras.Model): The loaded Keras model.

        '''        
        model_folders = []
        for entry in os.scandir(path):
            if entry.is_dir():
                model_folders.append(entry.name)
        if len(model_folders) > 1:
            model_folders.sort()
            index_list = [idx + 1 for idx, item in enumerate(model_folders)]     
            print('Please select a pretrained model:') 
            print()
            for i, directory in enumerate(model_folders):
                print(f'{i + 1} - {directory}')        
            print()               
            while True:
                try:
                    dir_index = int(input('Type the model index to select it: '))
                    print()
                except:
                    continue
                break                         
            while dir_index not in index_list:
                try:
                    dir_index = int(input('Input is not valid! Try again: '))
                    print()
                except:
                    continue
            self.folder_path = os.path.join(path, model_folders[dir_index - 1])

        elif len(model_folders) == 1:
            self.folder_path = os.path.join(path, model_folders[0])                 
        
        model_path = os.path.join(self.folder_path, 'model') 
        model = tf.keras.models.load_model(model_path)
        path = os.path.join(self.folder_path, 'model_parameters.json')
        with open(path, 'r') as f:
            configuration = json.load(f)               
        
        return model, configuration
    
    #--------------------------------------------------------------------------
    def images_loader(self, path, picture_shape=(244, 244, 3)):
        image = tf.io.read_file(path)
        rgb_image = tf.image.decode_image(image, channels=3)
        rgb_image = tf.image.resize(rgb_image, picture_shape[:-1])        
        rgb_image = rgb_image/255.0        

        return rgb_image 
    

# [VALIDATION OF PRETRAINED MODELS]
#==============================================================================
# Collection of methods for machine learning validation and model evaluation
#==============================================================================
class ModelValidation:

    def __init__(self, model):        
        self.model = model
    
    #-------------------------------------------------------------------------- 
    def visual_validation(self, real_images, predicted_images, name, path):          

        num_pics = len(real_images)
        fig_path = os.path.join(path, f'{name}.jpeg')
        fig, axs = plt.subplots(num_pics, 2, figsize=(4, num_pics * 2))
        for i, (real, pred) in enumerate(zip(real_images, predicted_images)):                         
            axs[i, 0].imshow(real)
            if i == 0:
                axs[i, 0].set_title('Original picture')
            axs[i, 0].axis('off')
            axs[i, 1].imshow(pred)
            if i == 0:
                axs[i, 1].set_title('Reconstructed picture')
            axs[i, 1].axis('off')
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches='tight', format='jpeg', dpi=400)        
        plt.close()
