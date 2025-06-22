import cv2
import numpy as np
import tensorflow as tf

   
# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
class TrainingDataLoader:

    def __init__(self, configuration, shuffle=True):
        self.img_shape = (128, 128)   
        self.num_channels = 3   
        self.augmentation = configuration.get('use_img_augmentation')
        self.batch_size = configuration.get('batch_size', 32)
        self.shuffle_samples = configuration.get('shuffle_size', 1024)
        self.configuration = configuration
        self.shuffle = shuffle  

    # load and preprocess a single image
    #--------------------------------------------------------------------------
    def load_image(self, path): 
        image = tf.io.read_file(path)
        rgb_image = tf.image.decode_image(
            image, channels=self.num_channels, expand_animations=False)        
        rgb_image = tf.image.resize(rgb_image, self.img_shape)        
        
        return rgb_image      
    
    # load and preprocess a single image
    #--------------------------------------------------------------------------
    def load_and_process_image(self, path): 
        rgb_image = self.load_image(path)
        rgb_image = self.image_normalization(rgb_image)
        rgb_image = self.image_augmentation(rgb_image) if self.augmentation else rgb_image        
        
        return rgb_image, rgb_image 

    # define method perform data augmentation    
    #--------------------------------------------------------------------------
    def image_normalization(self, image):
        normalize_image = image/255.0        
                
        return normalize_image         

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

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def compose_tensor_dataset(self, images, batch_size, buffer_size=tf.data.AUTOTUNE):        
        batch_size = self.batch_size if batch_size is None else batch_size
        dataset = tf.data.Dataset.from_tensor_slices(images)                
        dataset = dataset.map(
            self.load_and_process_image, num_parallel_calls=buffer_size)        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        dataset = dataset.shuffle(buffer_size=self.shuffle_samples) if self.shuffle else dataset 

        return dataset         
      
    #--------------------------------------------------------------------------
    def build_training_dataloader(self, train_data, validation_data, batch_size=None):       
        train_dataset = self.compose_tensor_dataset(train_data, batch_size)
        validation_dataset = self.compose_tensor_dataset(validation_data, batch_size)       

        return train_dataset, validation_dataset


###############################################################################
class InferenceDataLoader:

    def __init__(self, configuration):   
        self.img_shape = (128, 128)   
        self.num_channels = 3            
        self.buffer_size = tf.data.AUTOTUNE                  
        self.color_encoding = cv2.COLOR_BGR2RGB if self.num_channels==3 else cv2.COLOR_BGR2GRAY
        self.batch_size = configuration.get('batch_size', 32)
        self.configuration = configuration           

    # load and preprocess a single image
    #--------------------------------------------------------------------------
    def load_image(self, path): 
        image = tf.io.read_file(path)
        rgb_image = tf.image.decode_image(
            image, channels=self.num_channels, expand_animations=False)        
        rgb_image = tf.image.resize(rgb_image, self.img_shape)        
        
        return rgb_image      
    
    # load and preprocess a single image
    #--------------------------------------------------------------------------
    def load_and_process_image(self, path): 
        rgb_image = self.load_image(path)
        rgb_image = self.image_normalization(rgb_image)              
        
        return rgb_image, rgb_image 

    # define method perform data augmentation    
    #--------------------------------------------------------------------------
    def image_normalization(self, image):
        normalize_image = image/255.0        
                
        return normalize_image              

    #--------------------------------------------------------------------------
    def load_image_as_array(self, path, normalization=True):       
        image = cv2.imread(path)          
        image = cv2.cvtColor(image, self.color_encoding)
        image = np.asarray(
            cv2.resize(image, self.img_shape[:-1]), dtype=np.float32)            
        if normalization:
            image = image/255.0       

        return image  

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def compose_tensor_dataset(self, images, batch_size):         
        batch_size = self.batch_size if batch_size is None else batch_size
        dataset = tf.data.Dataset.from_tensor_slices(images)                
        dataset = dataset.map(
            self.load_and_process_image, num_parallel_calls=self.buffer_size)        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=self.buffer_size)
       
        return dataset         
      
    #--------------------------------------------------------------------------
    def build_inference_dataloader(self, data, batch_size=None):       
        dataset = self.compose_tensor_dataset(data, batch_size)             

        return dataset          
      
   








    