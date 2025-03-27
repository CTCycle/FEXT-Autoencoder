import numpy as np
import tensorflow as tf

from FEXT.commons.constants import CONFIG
from FEXT.commons.logger import logger
             

###############################################################################
class TrainingDataLoaderProcessor:

    def __init__(self, configuration):
        # set the input image shape as a fixed parameter         
        self.img_shape = (128, 128)   
        self.num_channels = 3 # RGB images   
        self.augmentation = configuration["dataset"]["IMG_AUGMENTATION"]
        self.configuration = configuration  

    # load and preprocess a single image
    #--------------------------------------------------------------------------
    def load_image(self, path, normalize=True): 
        # load images using tensorflow IO operations for efficiency       
        image = tf.io.read_file(path) # read image file
        # decode image as RGB and resize it to image input shae
        rgb_image = tf.image.decode_image(
            image, channels=self.num_channels, expand_animations=False)        
        rgb_image = tf.image.resize(rgb_image, self.img_shape)        
        # normalize image to [0, 1] range     
        rgb_image = rgb_image/255.0 if normalize else rgb_image 
        
        return rgb_image, rgb_image       
    
    # load and preprocess a single image
    #--------------------------------------------------------------------------
    def load_and_process_image(self, path): 
        # load images using tensorflow IO operations for efficiency       
        image = tf.io.read_file(path) # read image file
        # decode image as RGB and resize it to image input shae
        rgb_image = tf.image.decode_image(
            image, channels=self.num_channels, expand_animations=False)        
        rgb_image = tf.image.resize(rgb_image, self.img_shape)
        # apply image augmentation if requested
        rgb_image = self.image_augmentation(rgb_image) if self.augmentation else rgb_image  
        # normalize image to [0, 1] range     
        rgb_image = rgb_image/255.0 
        
        return rgb_image, rgb_image         

    # define method perform data augmentation    
    #--------------------------------------------------------------------------
    def image_augmentation(self, image):
        # perform random image augmentations such as flip, brightness, contrast          
        augmentations = {"flip_left_right": (
            lambda img: tf.image.random_flip_left_right(img), 0.5),
                         "flip_up_down": (
            lambda img: tf.image.random_flip_up_down(img), 0.5),                        
                         "brightness": (
            lambda img: tf.image.random_brightness(img, max_delta=0.2), 0.25),
                         "contrast": (
            lambda img: tf.image.random_contrast(img, lower=0.7, upper=1.3), 0.35)}    
        
        for _, (func, prob) in augmentations.items():
            if np.random.rand() <= prob:
                image = func(image)
        
        return image
   




            
