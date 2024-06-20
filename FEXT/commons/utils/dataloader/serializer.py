import os
import cv2
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import tensorflow as tf
from keras.utils import plot_model

from FEXT.commons.configurations import SAVE_MODEL_PLOT
from FEXT.commons.pathfinder import IMG_DATA_PATH, CHECKPOINT_PATH

    



#------------------------------------------------------------------------------
class DataSerializer:

    def __init__(self):
        
        self.model_name = 'FeXT'
        self.outputs = {'JSON' : self.JSON_serialization}

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

    # function to create a folder where to save model checkpoints
    #--------------------------------------------------------------------------
    def JSON_serialization(self, train_data, validation_data, test_data, path):

        combined_data = {'train': train_data, 
                         'validation': validation_data, 
                         'test': test_data}

        with open(os.path.join(path, 'preprocessed_data.json'), 'w') as json_file:
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

        self.preprocessing_path = os.path.join(checkpoint_folder_path, 'preprocessing')
        os.makedirs(self.preprocessing_path, exist_ok=True)
        
        return checkpoint_folder_path
    
    # function to create a folder where to save model checkpoints
    #--------------------------------------------------------------------------
    def save_preprocessed_data(self, train_data, validation_data, test_data, output_type='JSON'):

        if not self.preprocessing_path or not os.path.exists(self.preprocessing_path):
            self.create_checkpoint_folder()        

        # Serialize to selected output type
        self.outputs[output_type](train_data, validation_data, test_data, self.preprocessing_path)


#------------------------------------------------------------------------------
class ModelSerializer:

    def __init__(self):
        pass

    #--------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------
    def save_model_plot(self, model, path):

        if SAVE_MODEL_PLOT:
            plot_path = os.path.join(path, 'model_layout.png')       
            plot_model(model, to_file=plot_path, show_shapes=True, 
                    show_layer_names=True, show_layer_activations=True, 
                    expand_nested=True, rankdir='TB', dpi=400)
            
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
            self.loaded_model_folder = os.path.join(path, model_folders[dir_index - 1])

        elif len(model_folders) == 1:
            self.loaded_model_folder = os.path.join(path, model_folders[0])                 
        
        model_path = os.path.join(self.loaded_model_folder, 'model') 
        model = tf.keras.models.load_model(model_path)
        path = os.path.join(self.loaded_model_folder, 'model_parameters.json')
        with open(path, 'r') as f:
            configuration = json.load(f)               
        
        return model, configuration