import os
import sys
import cv2
import json
import pandas as pd
import keras
from datetime import datetime
import tensorflow as tf

from FEXT.commons.constants import CONFIG, CHECKPOINT_PATH
from FEXT.commons.logger import logger


# get the path of multiple images from a given directory
###############################################################################
def get_images_path(path, sample_size=None):
    
    if sample_size is None:
        sample_size =  CONFIG["dataset"]["SAMPLE_SIZE"]
        
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    logger.debug(f'Valid extensions are: {valid_extensions}')
    images_path = []
    for root, _, files in os.walk(path):
        if sample_size is not None:
            files = files[:int(sample_size*len(files))]           
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                images_path.append(os.path.join(root, file))                

    return images_path


# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:

    def __init__(self):        
        self.color_encoding = cv2.COLOR_BGR2RGB
        self.img_shape = CONFIG["model"]["IMG_SHAPE"]
        self.resized_img_shape = self.img_shape[:-1]
        self.normalization = CONFIG["dataset"]["IMG_NORMALIZE"]       
       
    #--------------------------------------------------------------------------
    def load_image(self, path, as_tensor=True):               
        
        if as_tensor:
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3)
            image = tf.image.resize(image, self.resized_img_shape)
            image = tf.reverse(image, axis=[-1])
            if self.normalization:
                image = image/255.0              
        else:
            image = cv2.imread(path)             
            image = cv2.resize(image, self.resized_img_shape)            
            image = cv2.cvtColor(image, self.color_encoding) 
            if self.normalization:
                image = image/255.0         
           

        return image

    # ...
    #--------------------------------------------------------------------------
    def save_preprocessed_data(self, train_data : list, validation_data : list, path):  
         
        train_dataframe = pd.DataFrame(train_data, columns=['image path'])
        validation_dataframe = pd.DataFrame(validation_data, columns=['image path'])       
        train_data_path = os.path.join(path, 'data', 'train_data.csv')
        val_data_path = os.path.join(path, 'data', 'validation_data.csv')
        train_dataframe.to_csv(train_data_path, index=False, sep=';', encoding='utf-8')
        validation_dataframe.to_csv(val_data_path, index=False, sep=';', encoding='utf-8')
        logger.debug(f'Preprocessed data has been saved at {path}')

    # ...
    #--------------------------------------------------------------------------
    def load_preprocessed_data(self, path):

        # Paths to the CSV files for train and validation data
        train_data_path = os.path.join(path, 'data', 'train_data.csv')
        val_data_path = os.path.join(path, 'data', 'validation_data.csv')

        # Check if the CSV files exist
        if not os.path.exists(train_data_path):
            logger.error(f'{train_data_path} does not exist.')
            return None, None
            
        if not os.path.exists(val_data_path):
            logger.error(f'{val_data_path} does not exist.')
            return None, None

        # Load the CSV files using pandas
        train_dataframe = pd.read_csv(train_data_path, sep=';', encoding='utf-8')
        validation_dataframe = pd.read_csv(val_data_path, sep=';', encoding='utf-8')

        # Extract image paths from the dataframes
        train_data = train_dataframe['image path'].tolist()
        validation_data = validation_dataframe['image path'].tolist()
        logger.debug(f'Preprocessed data has been loaded from {path}')

        return train_data, validation_data

    
# [MODEL SERIALIZATION]
###############################################################################
class ModelSerializer:

    def __init__(self):
        self.model_name = 'FeXT'

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
        checkpoint_folder_path = os.path.join(CHECKPOINT_PATH, f'{self.model_name}_{today_datetime}')         
        os.makedirs(checkpoint_folder_path, exist_ok=True)        
        os.makedirs(os.path.join(checkpoint_folder_path, 'data'), exist_ok=True)
        logger.debug(f'Created checkpoint folder at {checkpoint_folder_path}')
        
        return checkpoint_folder_path    

    #--------------------------------------------------------------------------
    def save_pretrained_model(self, model : keras.Model, path):

        model_files_path = os.path.join(path, 'saved_model.keras')
        model.save(model_files_path)
        logger.info(f'Training session is over. Model has been saved in folder {path}')

    #--------------------------------------------------------------------------
    def save_session_configuration(self, path, history : dict, configurations : dict):

        '''
        Saves the model parameters to a JSON file. The parameters are provided 
        as a dictionary and are written to a file named 'model_parameters.json' 
        in the specified directory.

        Keyword arguments:
            parameters_dict (dict): A dictionary containing the parameters to be saved.
            path (str): The directory path where the parameters will be saved.

        Returns:
            None  

        '''
        config_folder = os.path.join(path, 'configurations')
        os.makedirs(config_folder, exist_ok=True)

        # Paths to the JSON files
        config_path = os.path.join(config_folder, 'configurations.json')
        history_path = os.path.join(config_folder, 'session_history.json')

        # Function to merge dictionaries
        def merge_dicts(original, new_data):
            for key, value in new_data.items():
                if key in original:
                    if isinstance(value, dict) and isinstance(original[key], dict):
                        merge_dicts(original[key], value)
                    elif isinstance(value, list) and isinstance(original[key], list):
                        original[key].extend(value)
                    else:
                        original[key] = value
                else:
                    original[key] = value    

        # Save the merged configurations
        with open(config_path, 'w') as f:
            json.dump(configurations, f)

        # Load existing session history if the file exists and merge
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                existing_history = json.load(f)
            merge_dicts(existing_history, history)
        else:
            existing_history = history

        # Save the merged session history
        with open(history_path, 'w') as f:
            json.dump(existing_history, f)

        logger.debug(f'Model configuration and session history have been saved and merged at {path}')      

    #--------------------------------------------------------------------------
    def load_session_configuration(self, path): 

        config_path = os.path.join(path, 'configurations', 'configurations.json')        
        with open(config_path, 'r') as f:
            configurations = json.load(f)        

        history_path = os.path.join(path, 'configurations', 'session_history.json')
        with open(history_path, 'r') as f:
            history = json.load(f)

        return configurations, history

    #--------------------------------------------------------------------------
    def save_model_plot(self, model, path):

        if CONFIG["model"]["SAVE_MODEL_PLOT"]:
            logger.debug('Generating model architecture graph')
            plot_path = os.path.join(path, 'model_layout.png')       
            keras.utils.plot_model(model, to_file=plot_path, show_shapes=True, 
                       show_layer_names=True, show_layer_activations=True, 
                       expand_nested=True, rankdir='TB', dpi=400)
            
    #-------------------------------------------------------------------------- 
    def load_pretrained_model(self):

        '''
        Load a pretrained Keras model from the specified directory. If multiple model 
        directories are found, the user is prompted to select one. If only one model 
        directory is found, that model is loaded directly. If a 'model_parameters.json' 
        file is present in the selected directory, the function also loads the model 
        parameters.

        Keyword arguments:
            path (str): The directory path where the pretrained models are stored.
            load_parameters (bool, optional): If True, the function also loads the 
                                            model parameters from a JSON file. 
                                            Default is True.

        Returns:
            model (keras.Model): The loaded Keras model.
            configuration (dict): The loaded model parameters, or None if the parameters file is not found.

        '''  
        # look into checkpoint folder to get pretrained model names      
        model_folders = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():
                model_folders.append(entry.name)

        # quit the script if no pretrained models are found 
        if len(model_folders) == 0:
            logger.error('No pretrained model checkpoints in resources')
            sys.exit()

        # select model if multiple checkpoints are available
        if len(model_folders) > 1:
            model_folders.sort()
            index_list = [idx + 1 for idx, item in enumerate(model_folders)]     
            print('Currently available pretrained models:')             
            for i, directory in enumerate(model_folders):
                print(f'{i + 1} - {directory}')                         
            while True:
                try:
                    dir_index = int(input('\nSelect the pretrained model: '))
                    print()
                except ValueError:
                    logger.error('Invalid choice for the pretrained model, asking again')
                    continue
                if dir_index in index_list:
                    break
                else:
                    logger.warning('Model does not exist, please select a valid index')
                    
            self.loaded_model_folder = os.path.join(CHECKPOINT_PATH, model_folders[dir_index - 1])

        # load directly the pretrained model if only one is available 
        elif len(model_folders) == 1:
            logger.info('Loading pretrained model directly as only one is available')
            self.loaded_model_folder = os.path.join(CHECKPOINT_PATH, model_folders[0])                 
            
        # effectively load the model using keras builtin method
        model_path = os.path.join(self.loaded_model_folder, 'saved_model.keras') 
        model = keras.models.load_model(model_path)
        
        # load configuration data from .json file in checkpoint folder
        configuration, history = self.load_session_configuration(self.loaded_model_folder)        
            
        return model, configuration, history