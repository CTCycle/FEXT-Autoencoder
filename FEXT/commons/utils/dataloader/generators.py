import numpy as np
import tensorflow as tf

from FEXT.commons.constants import CONFIG
from FEXT.commons.logger import logger
             
        
# [CUSTOM DATA GENERATOR FOR TRAINING]
#------------------------------------------------------------------------------
# Generate batches of inputs and outputs using a custom generator function and 
# tf.dataset with prefetching
class DataGenerator():

    def __init__(self, data):       
              
        self.data = tf.convert_to_tensor(data)
        self.img_shape = CONFIG["model"]["IMG_SHAPE"]       
        self.normalization = CONFIG["dataset"]["IMG_NORMALIZE"]
        self.augmentation = CONFIG["dataset"]["IMG_AUGMENT"]    
    
    # define method to get X and Y data through custom functions, and subsequently
    # create a batch of data converted to tensors
    #--------------------------------------------------------------------------
    def load_image(self, path):
        image = tf.io.read_file(path)
        rgb_image = tf.image.decode_image(image, channels=3, expand_animations=False)        
        rgb_image = tf.image.resize(rgb_image, self.img_shape[:-1])
        if self.augmentation:
            rgb_image = self.image_augmentation(rgb_image)
        if self.normalization:
            rgb_image = rgb_image/255.0 

        return rgb_image 
    
    # define method to get X and Y data through custom functions, and subsequently
    # create a batch of data converted to tensors
    #--------------------------------------------------------------------------
    def process_data(self, path):

        rgb_image = self.load_image(path)        

        return rgb_image, rgb_image      

    # define method perform data augmentation    
    #--------------------------------------------------------------------------
    def image_augmentation(self, image):
        pp_image = tf.keras.preprocessing.image.random_shift(image, 0.2, 0.3)
        pp_image = tf.image.random_flip_left_right(pp_image)
        pp_image = tf.image.random_flip_up_down(pp_image)

        return pp_image 
              
        
# [CUSTOM DATA GENERATOR FOR TRAINING]
#------------------------------------------------------------------------------
def build_tensor_dataset(data, buffer_size=tf.data.AUTOTUNE):    
    
    num_samples = len(data)  
    batch_size = CONFIG["training"]["BATCH_SIZE"] 

    # initialize data generator tools
    datagen = DataGenerator(data)                    
    # create dataset from the list of images path, apply shuffling and map processing
    dataset = tf.data.Dataset.from_tensor_slices(data)    
    dataset = dataset.shuffle(buffer_size=num_samples)  
    dataset = dataset.map(datagen.process_data, num_parallel_calls=buffer_size)   
    # batch and prefetch dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=buffer_size)
    # logging debug info about batch shapes
    for x, y in dataset.take(1):
        logger.debug(f'X batch shape is: {x.shape}')  
        logger.debug(f'Y batch shape is: {y.shape}') 

    return dataset    
                
            
