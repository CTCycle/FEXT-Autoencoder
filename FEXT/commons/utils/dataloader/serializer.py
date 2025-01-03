import os
import sys
import cv2
import json
import numpy as np
import pandas as pd
import keras
from datetime import datetime

from FEXT.commons.constants import CONFIG, IMG_DATA_PATH, CHECKPOINT_PATH
from FEXT.commons.logger import logger


###############################################################################
def checkpoint_selection_menu(models_list):

    index_list = [idx + 1 for idx, item in enumerate(models_list)]     
    print('Currently available pretrained models:')             
    for i, directory in enumerate(models_list):
        print(f'{i + 1} - {directory}')                         
    while True:
        try:
            selection_index = int(input('\nSelect the pretrained model: '))
            print()
        except ValueError:
            logger.error('Invalid choice for the pretrained model, asking again')
            continue
        if selection_index in index_list:
            break
        else:
            logger.warning('Model does not exist, please select a valid index')

    return selection_index


# get the path of multiple images from a given directory
###############################################################################
def get_images_path(path, configuration=None, sample_size=None):
    
    if (sample_size is None and
        configuration is not None):
        sample_size = configuration["dataset"]["SAMPLE_SIZE"]
        
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

    def __init__(self, configuration):        
        self.color_encoding = cv2.COLOR_BGR2RGB
        self.img_shape = configuration["model"]["IMG_SHAPE"]             
        self.resized_img_shape = self.img_shape[:-1]      
       
    #--------------------------------------------------------------------------
    def load_image(self, path, normalization=True):       
        image = cv2.imread(path)          
        image = cv2.cvtColor(image, self.color_encoding)
        image = np.asarray(cv2.resize(image, self.img_shape[:-1]), dtype=np.float32)            
        if normalization:
            image = image / 255.0       

        return image   

    #--------------------------------------------------------------------------
    def save_preprocessed_data(self, train_data : list, validation_data : list, path):         
        
        train_dataframe = pd.DataFrame([os.path.basename(img) for img in train_data], columns=['image name'])
        validation_dataframe = pd.DataFrame([os.path.basename(img) for img in validation_data], columns=['image name'])          
        train_data_path = os.path.join(path, 'data', 'train_data.csv')
        val_data_path = os.path.join(path, 'data', 'validation_data.csv')
        train_dataframe.to_csv(train_data_path, index=False, sep=';', encoding='utf-8')
        validation_dataframe.to_csv(val_data_path, index=False, sep=';', encoding='utf-8')        

    #--------------------------------------------------------------------------
    def load_preprocessed_data(self, path):
        
        train_data_path = os.path.join(path, 'data', 'train_data.csv')
        val_data_path = os.path.join(path, 'data', 'validation_data.csv')

        if not os.path.exists(train_data_path):
            logger.error(f'{train_data_path} does not exist.')
            return None, None
            
        if not os.path.exists(val_data_path):
            logger.error(f'{val_data_path} does not exist.')
            return None, None
        
        train_dataframe = pd.read_csv(train_data_path, sep=';', encoding='utf-8')
        validation_dataframe = pd.read_csv(val_data_path, sep=';', encoding='utf-8')
        
        # load the list of image names and append the path to the image
        train_data = [os.path.join(IMG_DATA_PATH, img) 
                      for img in train_dataframe['image name'].tolist()]
        validation_data = [os.path.join(IMG_DATA_PATH, img)
                           for img in validation_dataframe['image name'].tolist()]
        
        return train_data, validation_data

    
# [MODEL SERIALIZATION]
###############################################################################
class ModelSerializer:

    def __init__(self):
        self.model_name = 'FeXT'        
        
    # function to create a folder where to save model checkpoints
    #--------------------------------------------------------------------------
    def create_checkpoint_folder(self):
   
        today_datetime = datetime.now().strftime('%Y%m%dT%H%M%S')        
        checkpoint_path = os.path.join(CHECKPOINT_PATH, f'{self.model_name}_{today_datetime}')         
        os.makedirs(checkpoint_path, exist_ok=True)        
        os.makedirs(os.path.join(checkpoint_path, 'data'), exist_ok=True)
        logger.debug(f'Created checkpoint folder at {checkpoint_path}')
        
        return checkpoint_path    

    #--------------------------------------------------------------------------
    def save_pretrained_model(self, model : keras.Model, path):
        model_files_path = os.path.join(path, 'saved_model.keras')
        model.save(model_files_path)
        logger.info(f'Training session is over. Model has been saved in folder {path}')

    #--------------------------------------------------------------------------
    def save_session_configuration(self, path, history : dict, configurations : dict):        
        config_folder = os.path.join(path, 'configurations')
        os.makedirs(config_folder, exist_ok=True)        
        config_path = os.path.join(config_folder, 'configurations.json')
        history_path = os.path.join(config_folder, 'session_history.json')        

        # Save the merged configurations
        with open(config_path, 'w') as f:
            json.dump(configurations, f)       

        # Save the merged session history
        with open(history_path, 'w') as f:
            json.dump(history, f)

        logger.debug(f'Model configuration and session history have been saved at {path}') 

    #-------------------------------------------------------------------------- 
    def scan_checkpoints_folder(self):
        model_folders = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():
                model_folders.append(entry.name)
        
        return model_folders     

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

        logger.debug(f'Plotting model architecture graph at {path}')
        plot_path = os.path.join(path, 'model_layout.png')       
        keras.utils.plot_model(model, to_file=plot_path, show_shapes=True, 
                    show_layer_names=True, show_layer_activations=True, 
                    expand_nested=True, rankdir='TB', dpi=400)
        
    #--------------------------------------------------------------------------
    def load_checkpoint(self, checkpoint_name):

        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint_name)
        model_path = os.path.join(checkpoint_path, 'saved_model.keras') 
        model = keras.models.load_model(model_path) 
        
        return model
            
    #-------------------------------------------------------------------------- 
    def select_and_load_checkpoint(self):
        
        # look into checkpoint folder to get pretrained model names      
        model_folders = self.scan_checkpoints_folder()

        # quit the script if no pretrained models are found 
        if len(model_folders) == 0:
            logger.error('No pretrained model checkpoints in resources')
            sys.exit()

        # select model if multiple checkpoints are available
        if len(model_folders) > 1:
            selection_index = checkpoint_selection_menu(model_folders)                    
            checkpoint_path = os.path.join(CHECKPOINT_PATH, model_folders[selection_index-1])

        # load directly the pretrained model if only one is available 
        elif len(model_folders) == 1:
            checkpoint_path = os.path.join(CHECKPOINT_PATH, model_folders[0])
            logger.info(f'Since only checkpoint {os.path.basename(checkpoint_path)} is available, it will be loaded directly')
                            
            
        # effectively load the model using keras builtin method
        # load configuration data from .json file in checkpoint folder
        model = self.load_checkpoint(checkpoint_path)       
        configuration, history = self.load_session_configuration(checkpoint_path)        
            
        return model, configuration, history, checkpoint_path