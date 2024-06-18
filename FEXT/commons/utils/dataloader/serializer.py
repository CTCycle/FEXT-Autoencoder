import os
import cv2
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import tensorflow as tf

from FEXT.commons.pathfinder import IMG_DATA_PATH, CHECKPOINT_PATH

    
#------------------------------------------------------------------------------
def load_images(paths, image_size, as_tensor=True, normalize=True):
        
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


#------------------------------------------------------------------------------
class DataSerializer:

    def __init__(self):
        
        self.model_name = 'FeXT'
        self.outputs = {'JSON' : self.JSON_serialization}

    # function to create a folder where to save model checkpoints
    #--------------------------------------------------------------------------
    def JSON_serialization(self, train_data, validation_data, test_data):

        combined_data = {'train': train_data, 
                         'validation': validation_data, 
                         'test': test_data}

        with open('preprocessed_data.json', 'w') as json_file:
            json.dump(combined_data, json_file)

        
    # function to create a folder where to save model checkpoints
    #--------------------------------------------------------------------------
    def create_checkpoint_folder(self):

        '''
        Creates a folder with the current date and time to save the model.

        Keyword arguments:
            None

        Returns:
            str: A string containing the path of the folder where the model will be saved.
        
        '''        
        today_datetime = datetime.now().strftime('%Y%m%dT%H%M%S')
        checkpoint_folder_name = f'{self.model_name}_{today_datetime}'
        checkpoint_folder_path = os.path.join(CHECKPOINT_PATH, checkpoint_folder_name)        
        # Create the directory if it does not exist
        os.makedirs(checkpoint_folder_path, exist_ok=True)

        self.preprocessing_path = os.path.join(checkpoint_folder_name, 'preprocessing')
        os.makedirs(self.preprocessing_path, exist_ok=True)
        
        return checkpoint_folder_path
    
    # function to create a folder where to save model checkpoints
    #--------------------------------------------------------------------------
    def save_preprocessed_data(self, train_data, validation_data, test_data, output_type='JSON'):

        if not self.preprocessing_path or not os.path.exists(self.preprocessing_path):
            self.create_checkpoint_folder()        

        # Serialize to selected output type
        self.outputs[output_type](train_data, validation_data, test_data)


#------------------------------------------------------------------------------
class ModelSerializer:

    def __init__(self):
        pass

        



    def save_model_parameters(self, parameters_dict):

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
        path = os.path.join(self.path, 'model_parameters.json')      
        with open(path, 'w') as f:
            json.dump(parameters_dict, f)