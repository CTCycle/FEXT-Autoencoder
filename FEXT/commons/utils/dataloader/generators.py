import numpy as np
import tensorflow as tf

from FEXT.commons.constants import CONFIG
from FEXT.commons.logger import logger
             
        
# [CUSTOM DATA GENERATOR]
###############################################################################
class DataGenerator:

    def __init__(self, configuration):        
        self.img_shape = configuration["model"]["IMG_SHAPE"]       
        self.normalization = configuration["dataset"]["IMG_NORMALIZE"]
        self.augmentation = configuration["dataset"]["IMG_AUGMENT"]
        self.configuration = configuration         
    
    # load and preprocess a single image
    #--------------------------------------------------------------------------
    def load_image(self, path):
        
        image = tf.io.read_file(path)
        rgb_image = tf.image.decode_image(image, channels=3, expand_animations=False)        
        rgb_image = tf.image.resize(rgb_image, self.img_shape[:-1])
        if self.augmentation:
            rgb_image = self.image_augmentation(rgb_image)
        if self.normalization:
            rgb_image = rgb_image/255.0 

        return rgb_image, rgb_image         

    # define method perform data augmentation    
    #--------------------------------------------------------------------------
    def image_augmentation(self, image):    
           
        augmentations = {"flip_left_right": (lambda img: tf.image.random_flip_left_right(img), 0.5),
                        "flip_up_down": (lambda img: tf.image.random_flip_up_down(img), 0.5),                        
                        "brightness": (lambda img: tf.image.random_brightness(img, max_delta=0.2), 0.25),
                        "contrast": (lambda img: tf.image.random_contrast(img, lower=0.7, upper=1.3), 0.35)}    
        
        for _, (func, prob) in augmentations.items():
            if np.random.rand() <= prob:
                image = func(image)
        
        return image
              
    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #------------------------------------------------------------------------------
    def build_tensor_dataset(self, data, batch_size=None, buffer_size=tf.data.AUTOTUNE):

        '''
        Builds a TensorFlow dataset and applies preprocessing, batching, and prefetching.

        Keyword arguments:
            data (list): A list of image file paths.
            buffer_size (int): The buffer size for shuffling and prefetching (default is tf.data.AUTOTUNE).

        Returns:
            dataset (tf.data.Dataset): The prepared TensorFlow dataset.

        '''
        num_samples = len(data) 
        if batch_size is None:
            batch_size = self.configuration["training"]["BATCH_SIZE"]

        dataset = tf.data.Dataset.from_tensor_slices(data)                  
        dataset = dataset.map(self.load_image, num_parallel_calls=buffer_size)        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        dataset = dataset.shuffle(buffer_size=num_samples)

        return dataset

    
# LAUNCHER function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
def training_data_pipeline(train_data, validation_data, configuration, batch_size=None):    
        
        generator = DataGenerator(configuration)
        train_dataset = generator.build_tensor_dataset(train_data, batch_size=batch_size)
        validation_dataset = generator.build_tensor_dataset(validation_data, batch_size=batch_size)        
        for x, y in train_dataset.take(1):
            logger.debug(f'X batch shape is: {x.shape}')  
            logger.debug(f'Y batch shape is: {y.shape}') 

        return train_dataset, validation_dataset



            
