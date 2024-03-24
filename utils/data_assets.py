import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import tensorflow as tf
from tensorflow import keras


# [PREPROCESSING PIPELINE]
#==============================================================================
# Series of methods and functions to preprocess data for model training
#==============================================================================
class PreProcessing:    

    #--------------------------------------------------------------------------
    def dataset_from_images(self, path, dataset=None):

        '''
        Add a column with relative path to images in a dataframe, given a column where the
        images names are stored
    
        Keyword arguments:
            path (str):         A string containing the path where the images are located
            dataframe (pandas): the selected dataframe
            id_col (str):       Name of the column with images names 
    
        Returns:
            dataframe: the modified dataframe
        
        '''
        if dataset is None:
            image_locations = []
            image_names = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    image_locations.append(os.path.join(root, file))
                    image_names.append(file)            
            dataset = pd.DataFrame({'name': image_names, 'path': image_locations})  
        else:      
            dataset['path'] = dataset['name'].apply(lambda x : os.path.join(path, x))  

        return dataset

    #--------------------------------------------------------------------------
    def load_images(self, paths, image_size, as_tensor=True, normalize=True):
        
        images = []
        for pt in tqdm(paths):
            if as_tensor==False:                
                image = cv2.imread(pt)             
                image = cv2.resize(image, image_size)            
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                if normalize==True:
                    image = image/255.0
            else:
                image = tf.io.read_file(pt)
                image = tf.image.decode_image(image, channels=3)
                image = tf.image.resize(image, image_size)
                image = tf.reverse(image, axis=[-1])
                if normalize==True:
                    image = image/255.0
            
            images.append(image) 

        return images    
    
    #--------------------------------------------------------------------------
    def model_savefolder(self, path, model_name):

        '''
        Creates a folder with the current date and time to save the model.
    
        Keyword arguments:
            path (str):       A string containing the path where the folder will be created.
            model_name (str): A string containing the name of the model.
    
        Returns:
            str: A string containing the path of the folder where the model will be saved.
        
        '''        
        today_datetime = str(datetime.now())
        truncated_datetime = today_datetime[:-10]
        today_datetime = truncated_datetime.replace(':', '').replace('-', '').replace(' ', 'H') 
        self.folder_name = f'{model_name}_{today_datetime}'
        model_folder_path = os.path.join(path, self.folder_name)
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path) 
                    
        return model_folder_path 
    

# [CUSTOM DATA GENERATOR FOR TRAINING]
#==============================================================================
# Generate batches of inputs and outputs using a custom generator function and 
# tf.dataset with prefetching
#==============================================================================
class DataGenerator(keras.utils.Sequence):

    def __init__(self, dataframe, batch_size, picture_shape=(244, 244, 3), shuffle=True,
                  augmentation=True, normalization=True):        
        self.dataframe = dataframe
        self.path_col = 'path'       
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
        
    
# [TF.DATASET GENERATION]
#==============================================================================
# Generate batches of inputs and outputs using a custom generator function and 
# tf.dataset with prefetching
#==============================================================================
class TensorDataSet():

    
    # create tensorflow dataset from generator    
    #--------------------------------------------------------------------------
    def create_tf_dataset(self, generator, buffer_size=tf.data.AUTOTUNE):
        
        x_batch, y_batch = generator.__getitem__(0)        
        output_signature = (tf.TensorSpec(shape=x_batch.shape, dtype=tf.float32), 
                            tf.TensorSpec(shape=y_batch.shape, dtype=tf.float32))
        dataset = tf.data.Dataset.from_generator(lambda : generator, output_signature=output_signature)
        dataset = dataset.prefetch(buffer_size=buffer_size) 

        return dataset
              

# [VALIDATION OF DATA]
#==============================================================================
# Series of methods and functions to preprocess data for model training
#==============================================================================
class DataValidation:

    def pixel_intensity_histograms(self, image_set_1, image_set_2, path, params,
                                   names=['First set', 'Second set']):
        
        pixel_intensities_1 = np.concatenate([image.flatten() for image in image_set_1])
        pixel_intensities_2 = np.concatenate([image.flatten() for image in image_set_2])        
        plt.hist(pixel_intensities_1, bins='auto', alpha=0.5, color='blue', label=names[0])
        plt.hist(pixel_intensities_2, bins='auto', alpha=0.5, color='red', label=names[1])
        plt.title(params['title'],)
        plt.xlabel('Pixel Intensity', fontsize=params['fontsize_labels'])
        plt.ylabel(params['ylabel'],  fontsize=params['fontsize_labels'])
        plt.legend()            
        plt.tight_layout()
        plot_loc = os.path.join(path, params['filename'])
        plt.savefig(plot_loc, bbox_inches='tight', format='jpeg', dpi=400)   
        plt.show()         
        plt.close()
        
              
        
        
            
        
