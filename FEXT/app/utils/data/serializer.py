import os
import json

import pandas as pd
from keras.utils import plot_model
from keras.models import load_model
from datetime import datetime

from FEXT.app.utils.data.database import FEXTDatabase
from FEXT.app.utils.learning.training.scheduler import LinearDecayLRScheduler
from FEXT.app.constants import CHECKPOINT_PATH
from FEXT.app.logger import logger


# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:

    def __init__(self, configuration):        
        self.img_shape = (128, 128, 3)
        self.num_channels = self.img_shape[-1] 
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}        
        self.seed = configuration.get('general_seed', 42)
        self.configuration = configuration
        self.database = FEXTDatabase()
    
    #--------------------------------------------------------------------------
    def get_images_path_from_directory(self, path : str, sample_size=1.0):            
        logger.debug(f'Valid extensions are: {self.valid_extensions}')
        images_path = []
        for root, _, files in os.walk(path):
            if sample_size < 1.0:
                files = files[:int(sample_size * len(files))]           
            for file in files:                
                if os.path.splitext(file)[1].lower() in self.valid_extensions:
                    images_path.append(os.path.join(root, file))                

        return images_path 
    
    #--------------------------------------------------------------------------
    def save_image_statistics(self, data : pd.DataFrame):            
        self.database.save_image_statistics(data)       
    
    #--------------------------------------------------------------------------
    def save_checkpoints_summary(self, data : pd.DataFrame):            
        self.database.save_checkpoints_summary(data)         
    
    
# [MODEL SERIALIZATION]
###############################################################################
class ModelSerializer:

    def __init__(self):        
        self.model_name = 'FeXT_AE'        
        
    # function to create a folder where to save model checkpoints
    #--------------------------------------------------------------------------
    def create_checkpoint_folder(self):   
        today_datetime = datetime.now().strftime('%Y%m%dT%H%M%S')        
        checkpoint_path = os.path.join(
            CHECKPOINT_PATH, f'{self.model_name}_{today_datetime}')         
        os.makedirs(checkpoint_path, exist_ok=True)  
        logger.debug(f'Created checkpoint folder at {checkpoint_path}')
        
        return checkpoint_path    

    #--------------------------------------------------------------------------
    def save_pretrained_model(self, model, path):
        model_files_path = os.path.join(path, 'saved_model.keras')
        model.save(model_files_path)
        logger.info(f'Training session is over. Model {os.path.basename(path)} has been saved')

    #--------------------------------------------------------------------------
    def save_training_configuration(self, path, session, configuration : dict):         
        os.makedirs(os.path.join(path, 'configuration'), exist_ok=True)        
        config_path = os.path.join(path, 'configuration', 'configuration.json')
        history_path = os.path.join(path, 'configuration', 'session_history.json')
        history = {'history' : session.history,
                   'epochs': session.epoch[-1] + 1}

        # Save training and model configuration
        with open(config_path, 'w') as f:
            json.dump(configuration, f)
            
        # Save session history
        with open(history_path, 'w') as f:
            json.dump(history, f)
        
    #-------------------------------------------------------------------------- 
    def scan_checkpoints_folder(self):
        model_folders = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():
                model_folders.append(entry.name)
        
        return model_folders     

    #--------------------------------------------------------------------------
    def load_training_configuration(self, path):
        config_path = os.path.join(
            path, 'configuration', 'configuration.json')        
        with open(config_path, 'r') as f:
            configuration = json.load(f)        

        history_path = os.path.join(
            path, 'configuration', 'session_history.json')
        with open(history_path, 'r') as f:
            history = json.load(f)

        return configuration, history

    #--------------------------------------------------------------------------
    def save_model_plot(self, model, path):
        logger.debug(f'Plotting model architecture graph at {path}')
        plot_path = os.path.join(path, 'model_layout.png')       
        plot_model(
            model, to_file=plot_path, show_shapes=True, show_layer_names=True, 
            show_layer_activations=True, expand_nested=True, rankdir='TB', dpi=400)        
            
    #-------------------------------------------------------------------------- 
    def load_checkpoint(self, checkpoint_name : str):                     
        # effectively load the model using keras builtin method
        # load configuration data from .json file in checkpoint folder
        custom_objects = {'LinearDecayLRScheduler': LinearDecayLRScheduler}                
        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint_name) 
        model_path = os.path.join(checkpoint_path, 'saved_model.keras') 
        model = load_model(model_path, custom_objects=custom_objects)       
        configuration, session = self.load_training_configuration(checkpoint_path)        
            
        return model, configuration, session, checkpoint_path