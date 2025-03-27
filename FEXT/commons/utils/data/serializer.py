import os
import sys
import json
import numpy as np
import keras
from datetime import datetime

from FEXT.commons.utils.learning.scheduler import LRScheduler
from FEXT.commons.constants import CONFIG, CHECKPOINT_PATH
from FEXT.commons.logger import logger


###############################################################################
def checkpoint_selection_menu(models_list):
    # display an interactive menu to select a pretrained model from a numbered list
    # uses all checkpoints found in resources/checkpoints
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


# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:

    def __init__(self, configuration):        
        self.img_shape = (128, 128, 3)
        self.num_channels = self.img_shape[-1] 
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}        
        self.seed = configuration['SEED']   
        self.parameters = configuration["dataset"]
        self.configuration = configuration

    # get images path from a given directory 
    #--------------------------------------------------------------------------
    def get_images_path(self, path, sample_size=None): 
        # get sample size reduction from configurations if not directly provided
        sample_size = self.parameters["SAMPLE_SIZE"] if sample_size is None else sample_size       
        logger.debug(f'Valid extensions are: {self.valid_extensions}')
        images_path = []
        for root, _, files in os.walk(path):
            if sample_size is not None:
                files = files[:int(sample_size*len(files))]           
            for file in files:
                # only consider files with valid image extensions
                if os.path.splitext(file)[1].lower() in self.valid_extensions:
                    images_path.append(os.path.join(root, file))                

        return images_path      
               
    
    
# [MODEL SERIALIZATION]
###############################################################################
class ModelSerializer:

    def __init__(self):
        self.model_name = 'FeXT'        
        
    # function to create a folder where to save model checkpoints
    #--------------------------------------------------------------------------
    def create_checkpoint_folder(self):   
        today_datetime = datetime.now().strftime('%Y%m%dT%H%M%S')        
        checkpoint_path = os.path.join(
            CHECKPOINT_PATH, f'{self.model_name}_{today_datetime}')         
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
        os.makedirs(os.path.join(path, 'configurations'), exist_ok=True)        
        config_path = os.path.join(path, 'configurations', 'configurations.json')
        history_path = os.path.join(path, 'configurations', 'session_history.json')

        # Save training and model configurations
        with open(config_path, 'w') as f:
            json.dump(configurations, f)
            
        # Save session history
        with open(history_path, 'w') as f:
            json.dump(history, f)

        logger.debug(f'Model configuration and session history saved for {os.path.basename(path)}') 

    #-------------------------------------------------------------------------- 
    def scan_checkpoints_folder(self):
        model_folders = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():
                model_folders.append(entry.name)
        
        return model_folders     

    #--------------------------------------------------------------------------
    def load_session_configuration(self, path):
        config_path = os.path.join(
            path, 'configurations', 'configurations.json')        
        with open(config_path, 'r') as f:
            configurations = json.load(f)        

        history_path = os.path.join(
            path, 'configurations', 'session_history.json')
        with open(history_path, 'r') as f:
            history = json.load(f)

        return configurations, history

    #--------------------------------------------------------------------------
    def save_model_plot(self, model, path):
        logger.debug(f'Plotting model architecture graph at {path}')
        plot_path = os.path.join(path, 'model_layout.png')       
        keras.utils.plot_model(
            model, to_file=plot_path, show_shapes=True, show_layer_names=True, 
            show_layer_activations=True, expand_nested=True, rankdir='TB', dpi=400)
        
    #--------------------------------------------------------------------------
    def load_checkpoint(self, checkpoint_name):
        custom_objects = {'LRScheduler': LRScheduler}                
        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint_name)
        model_path = os.path.join(checkpoint_path, 'saved_model.keras') 
        model = keras.models.load_model(model_path, custom_objects=custom_objects) 
        
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
            checkpoint_path = os.path.join(
                CHECKPOINT_PATH, model_folders[selection_index-1])

        # load directly the pretrained model if only one is available 
        elif len(model_folders) == 1:
            checkpoint_path = os.path.join(CHECKPOINT_PATH, model_folders[0])
            logger.info(f'Since only checkpoint {os.path.basename(checkpoint_path)} is available, it will be loaded directly')
           
        # effectively load the model using keras builtin method
        # load configuration data from .json file in checkpoint folder
        model = self.load_checkpoint(checkpoint_path)       
        configuration, history = self.load_session_configuration(checkpoint_path)        
            
        return model, configuration, history, checkpoint_path