import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D

    
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
            

# [TOOLS FOR TRAINING MACHINE LEARNING MODELS]
#==============================================================================
# collection of model and submodels
#==============================================================================
class AutoEncoderModel:

    def __init__(self, learning_rate, picture_size=(144, 144, 3), XLA_state=False):         
        self.learning_rate = learning_rate
        self.num_channels = 3
        self.picture_size = picture_size + (self.num_channels,)         
        self.XLA_state = XLA_state     

    
    #-------------------------------------------------------------------------- 
    def FEXT_encoder(self):
                 
        image_input = Input(shape = self.picture_size, name = 'image_input')
        #----------------------------------------------------------------------
        layer = Conv2D(64, (4,4), strides=1, padding = 'same', activation = 'relu')(image_input)          
        layer = Conv2D(64, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)                    
        layer = MaxPooling2D((2,2), padding = 'same')(layer) 
        #----------------------------------------------------------------------
        layer = Conv2D(128, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)           
        layer = Conv2D(128, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)                   
        layer = MaxPooling2D((2,2), padding = 'same')(layer)        
        #---------------------------------------------------------------------- 
        layer = Conv2D(256, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)       
        layer = Conv2D(256, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)       
        layer = Conv2D(256, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)        
        layer = MaxPooling2D((2,2), padding = 'same')(layer)
        #----------------------------------------------------------------------
        layer = Conv2D(512, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)       
        layer = Conv2D(512, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)        
        layer = Conv2D(512, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)        
        layer = MaxPooling2D((2,2), padding = 'same')(layer) 
        #----------------------------------------------------------------------
        layer = Conv2D(512, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)        
        layer = Conv2D(512, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)       
        layer = Conv2D(512, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)        
        layer = MaxPooling2D((2,2), padding = 'same')(layer)     
        #----------------------------------------------------------------------        
        output = Dense(512, activation = 'tanh')(layer)  
       
        self.encoder = Model(inputs = image_input, outputs = output, name = 'FeatEXT_encoder') 

        return self.encoder   
   

    
    #--------------------------------------------------------------------------
    def FEXT_decoder(self):        
        
        vector_input = Input(shape = self.encoder.layers[-1].output_shape[1:]) 
        #----------------------------------------------------------------------
        layer = UpSampling2D(size = (2, 2), interpolation='bilinear')(vector_input)
        #----------------------------------------------------------------------
        layer = Conv2DTranspose(512, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)        
        layer = Conv2DTranspose(512, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)        
        layer = Conv2DTranspose(512, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)                       
        layer = UpSampling2D(size = (2, 2), interpolation='bilinear')(layer)
        #----------------------------------------------------------------------
        layer = Conv2DTranspose(512, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)       
        layer = Conv2DTranspose(512, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)        
        layer = Conv2DTranspose(512, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)          
        layer = UpSampling2D(size = (2, 2), interpolation='bilinear')(layer)
        #----------------------------------------------------------------------
        layer = Conv2DTranspose(256, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)        
        layer = Conv2DTranspose(256, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)        
        layer = Conv2DTranspose(256, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)        
        layer = UpSampling2D(size = (2, 2), interpolation='bilinear')(layer)
        #----------------------------------------------------------------------
        layer = Conv2DTranspose(128, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)        
        layer = Conv2DTranspose(128, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)        
        layer = UpSampling2D(size = (2, 2), interpolation='bilinear')(layer)              
        #----------------------------------------------------------------------
        layer = Conv2DTranspose(64, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)         
        layer = Conv2DTranspose(64, (4,4), strides=1, padding = 'same', activation = 'relu')(layer)
        #----------------------------------------------------------------------         
        output = Dense(self.num_channels, activation = 'sigmoid', dtype='float32')(layer)              
        
        self.decoder = Model(inputs = vector_input, outputs = output, name = 'FeatEXT_decoder')

        return self.decoder      
    
    #-------------------------------------------------------------------------- 
    def FEXT_AutoEncoder(self):

        encoder = self.FEXT_encoder()
        decoder = self.FEXT_decoder() 
       
        image_input = Input(shape = self.picture_size)         
        #----------------------------------------------------------------------
        encoder_block = encoder(image_input)        
        decoder_block = decoder(encoder_block)
        #----------------------------------------------------------------------
        self.model = Model(inputs = image_input, outputs = decoder_block, name = 'FEXT_model')
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = keras.losses.MeanSquaredError()
        metric = keras.metrics.CosineSimilarity()
        self.model.compile(loss = loss, optimizer = opt, metrics = metric, 
                           run_eagerly=False, jit_compile=self.XLA_state) 

        return self.model
    

# [CUSTOM DATA GENERATOR FOR TRAINING]
#==============================================================================
# Generate batches of inputs and outputs using a custom generator function and 
# tf.dataset with prefetching
#==============================================================================
class DataGenerator(keras.utils.Sequence):

    def __init__(self, dataframe, batch_size, image_size=(244, 244), shuffle=True, augmentation=True):        
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
        rgb_image = tf.image.resize(rgb_image, self.image_size)        
        if self.augmentation==True:
            rgb_image = self.__images_augmentation(rgb_image)
        rgb_image = rgb_image / 255.0        

        return rgb_image    
    
    # define method to call the next elements of the generator    
    #--------------------------------------------------------------------------
    def next(self):
        next_index = (self.batch_index + 1) % self.__len__()
        self.batch_index = next_index

        return self.__getitem__(next_index)


# [TOOLS FOR TRAINING MACHINE LEARNING MODELS]
#==============================================================================
# Collection of methods for machine learning training and tensorflow settings
#==============================================================================
class ModelTraining:    
       
    def __init__(self, device = 'default', seed=42, use_mixed_precision=False):                     
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
        path = os.path.join(savepath, 'model_parameters.txt')      
        with open(path, 'w') as f:
            for key, value in parameters_dict.items():
                f.write(f'{key}: {value}\n') 

    #-------------------------------------------------------------------------- 
    def load_pretrained_model(self, path):
        
        model_folders = []
        for entry in os.scandir(path):
            if entry.is_dir():
                model_folders.append(entry.name)
        model_folders.sort()
        index_list = [idx + 1 for idx, item in enumerate(model_folders)]     
        print('Please select a pretrained model:') 
        print()
        for i, directory in enumerate(model_folders):
            print('{0} - {1}'.format(i + 1, directory))
        
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
        model = keras.models.load_model(self.model_path)        
        
        return model
        

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
