import os
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D

    
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
        if epoch % 5 == 0:                    
            self.epochs.append(epoch)
            self.loss_hist.append(logs[list(logs.keys())[0]])
            self.metric_hist.append(logs[list(logs.keys())[1]])
            if self.validation==True:
                self.loss_val_hist.append(logs[list(logs.keys())[2]])            
                self.metric_val_hist.append(logs[list(logs.keys())[3]])
        if epoch % 10 == 0:           
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
            plt.plot(self.epochs, self.metric_hist, label = 'train metrics') 
            if self.validation==True: 
                plt.plot(self.epochs, self.metric_val_hist, label = 'validation metrics') 
                plt.legend(loc = 'best', fontsize = 8)
            plt.title('metrics plot')
            plt.ylabel('Accuracy')
            plt.xlabel('epoch')       
            plt.tight_layout()
            plt.savefig(fig_path, bbox_inches = 'tight', format = 'jpeg', dpi = 300)            
            plt.close() 
            
# [CUSTOM DATA GENERATOR FOR TRAINING]
#==============================================================================
# Generate batches of inputs and outputs using a custom generator function and 
# tf.dataset with prefetching
#==============================================================================
class DataGenerator(keras.utils.Sequence):

    def __init__(self, dataframe, batch_size, image_size=(244, 244), shuffle=True,
                  augmentation=True):        
        self.dataframe = dataframe
        self.path_col = 'images path'       
        self.num_of_samples = dataframe.shape[0]
        self.image_size = image_size
        self.batch_size = batch_size  
        self.batch_index = 0 
        self.augmentation = augmentation             
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
        x1_batch = [self.__images_generation(image_path, augmentation=self.augmentation) for image_path in path_batch]        
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
    def __images_generation(self, path, augmentation=True):
        image = tf.io.read_file(path)
        rgb_image = tf.image.decode_image(image, channels=3)
        rgb_image = tf.image.resize(rgb_image, self.image_size)        
        if augmentation==True:
            rgb_image = self.__images_augmentation(rgb_image)
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
class PooledConvBlock(keras.layers.Layer):
    def __init__(self, units, kernel_size, layers=2, seed=42):
        super(PooledConvBlock, self).__init__()
        self.units = units
        self.kernel_size = kernel_size
        self.layers = layers
        self.seed = seed
        self.convolutions = [Conv2D(units, kernel_size=kernel_size, padding='same', activation='relu') for x in range(layers)]         
        self.pooling = MaxPooling2D(padding='same') 
        
        
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training):
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
                       'layers': self.layers,
                       'seed': self.seed})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)  

# [POOLING CONVOLUTIONAL BLOCKS]
#==============================================================================
# Positional embedding custom layer
#==============================================================================
class TransposeConvBlock(keras.layers.Layer):
    def __init__(self, units, kernel_size, layers=2, seed=42):
        super(TransposeConvBlock, self).__init__()
        self.units = units
        self.kernel_size = kernel_size
        self.layers = layers
        self.seed = seed
        self.convolutions = [Conv2DTranspose(units, kernel_size=kernel_size, padding='same', activation='relu') for x in range(layers)]         
        self.upsamp = UpSampling2D()         
        
    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training):
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
                       'layers': self.layers,
                       'seed': self.seed})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)      

    

       
# [MACHINE LEARNING MODELS]
#==============================================================================
# collection of model and submodels
#==============================================================================
class FeXTEncoder(keras.layers.Layer):
    def __init__(self, kernel_size, picture_size=(144, 144), seed=42):
        super(FeXTEncoder, self).__init__()
        self.kernel_size = kernel_size
        self.seed = seed
        self.num_channels = 3
        self.picture_shape = picture_size + (self.num_channels,)
        self.convblock1 = PooledConvBlock(64, kernel_size, 2, seed)
        self.convblock2 = PooledConvBlock(128, kernel_size, 2, seed)
        self.convblock3 = PooledConvBlock(256, kernel_size, 3, seed)
        self.convblock4 = PooledConvBlock(512, kernel_size, 3, seed)

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training=None):
        layer = inputs
        layer = self.convblock1(layer)
        layer = self.convblock2(layer)
        layer = self.convblock3(layer)
        output = self.convblock4(layer)
        return output

    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(FeXTEncoder, self).get_config()
        config.update({'kernel_size': self.kernel_size,
                       'seed': self.seed,
                       'num_channels': self.num_channels,
                       'picture_shape': self.picture_shape})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config) 

# [MACHINE LEARNING MODELS]
#==============================================================================
# collection of model and submodels
#==============================================================================
class FeXTDecoder(keras.layers.Layer):
    def __init__(self, kernel_size, picture_size=(144, 144), seed=42):
        super(FeXTDecoder, self).__init__()
        self.kernel_size = kernel_size
        self.seed = seed
        self.num_channels = 3
        self.picture_shape = picture_size + (self.num_channels,)
        self.convblock1 = TransposeConvBlock(512, kernel_size, 3, seed)
        self.convblock2 = TransposeConvBlock(256, kernel_size, 3, seed)
        self.convblock3 = TransposeConvBlock(128, kernel_size, 2, seed)
        self.convblock4 = TransposeConvBlock(64, kernel_size, 2, seed)
        self.dense = Dense(3, activation='sigmoid', dtype='float32')

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training):
        layer = inputs
        layer = self.convblock1(layer)
        layer = self.convblock2(layer)
        layer = self.convblock3(layer)
        layer = self.convblock4(layer)
        output = self.dense(layer)

        return output

    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(FeXTDecoder, self).get_config()
        config.update({'kernel_size': self.kernel_size,
                       'seed': self.seed,
                       'num_channels': self.num_channels,
                       'picture_shape': self.picture_shape})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)  
        
    

# [MACHINE LEARNING MODELS]
#==============================================================================
# collection of model and submodels
#==============================================================================
class FeXTAutoEncoder: 

    def __init__(self, learning_rate, kernel_size, picture_size=(144, 144), seed=42, 
                 XLA_state=False):         
        self.learning_rate = learning_rate
        self.kernel_size = kernel_size
        self.seed = seed
        self.num_channels = 3
        self.picture_shape = picture_size + (self.num_channels,)         
        self.XLA_state = XLA_state
        self.encoder = FeXTEncoder(kernel_size, picture_size, seed)
        self.decoder = FeXTDecoder(kernel_size, picture_size, seed)
         
        

    # build model given the architecture
    #--------------------------------------------------------------------------
    def build(self):       
       
        inputs = Input(shape = self.picture_shape)           
        encoder_block = self.encoder(inputs)        
        decoder_block = self.decoder(encoder_block)
        
        self.model = Model(inputs = inputs, outputs=decoder_block, name='FEXT_model')
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = keras.losses.MeanSquaredError()
        metric = keras.metrics.CosineSimilarity()
        self.model.compile(loss = loss, optimizer = opt, metrics = metric, 
                           run_eagerly=False, jit_compile=self.XLA_state) 

        return self.model  




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
            self.model_path = os.path.join(path, model_folders[dir_index - 1])

        elif len(model_folders) == 1:
            self.model_path = os.path.join(path, model_folders[0])                 
        
        model = keras.models.load_model(self.model_path)
        path = os.path.join(self.model_path, 'model_parameters.json')
        with open(path, 'r') as f:
            self.model_configuration = json.load(f)            
        
        return model
    
    #--------------------------------------------------------------------------
    def images_loader(self, path, image_size=(244, 244), num_channels=3):
        image = tf.io.read_file(path)
        rgb_image = tf.image.decode_image(image, channels=num_channels)
        rgb_image = tf.image.resize(rgb_image, image_size)        
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
    def FEXT_validation(self, real_images, predicted_images, name, path):
        
        num_pics = len(real_images)
        fig_path = os.path.join(path, f'{name}_validation.jpeg')
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
        plt.show(block=False)
        plt.close()
