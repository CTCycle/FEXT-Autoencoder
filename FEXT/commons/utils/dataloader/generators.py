import numpy as np
import tensorflow as tf

from FEXT.commons.constants import CONFIG
from FEXT.commons.logger import logger
             
        
# [CUSTOM DATA GENERATOR]
###############################################################################
class DatasetGenerator:

    def __init__(self, configuration):        
        self.img_shape = (128, 128)   
        self.num_channels = 3    
        self.augmentation = configuration["dataset"]["IMG_AUGMENT"]
        self.configuration = configuration         
    
    # load and preprocess a single image
    #--------------------------------------------------------------------------
    def load_image(self, path):        
        image = tf.io.read_file(path)
        rgb_image = tf.image.decode_image(
            image, channels=self.num_channels, expand_animations=False)        
        rgb_image = tf.image.resize(rgb_image, self.img_shape)
        rgb_image = self.image_augmentation(rgb_image) if self.augmentation else rgb_image       
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
   




            
