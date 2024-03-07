import os
import numpy as np
import json
import tensorflow as tf
from IPython.display import display
from ipywidgets import Dropdown

        
# [INFERENCE]
#==============================================================================
# Collection of methods for machine learning validation and model evaluation
#==============================================================================
class Inference:

    def __init__(self, seed):
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)  

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
        model_folders = [entry.name for entry in os.scandir(path) if entry.is_dir()]
    
        if len(model_folders) > 1:
            model_folders.sort()
            dropdown = Dropdown(options=model_folders, description='Select Model:')
            display(dropdown)
            # Wait for the user to select a model. This cell should be manually executed again after selection.            
            self.folder_path = os.path.join(path, dropdown.value)

        elif len(model_folders) == 1:
            self.folder_path = os.path.join(path, model_folders[0])
        else:
            raise FileNotFoundError('No model directories found in the specified path.')
        
        model_path = os.path.join(self.folder_path, 'model')
        model = tf.keras.models.load_model(model_path)
        
        configuration = {}        
        parameters_path = os.path.join(self.folder_path, 'model_parameters.json')
        if os.path.exists(parameters_path):
            with open(parameters_path, 'r') as f:
                configuration = json.load(f)
        else:
            print('No parameters file found. Continuing without loading parameters.')
            
        return model, configuration  
    
    #--------------------------------------------------------------------------
    def images_loader(self, path, picture_shape=(244, 244, 3)):
        image = tf.io.read_file(path)
        rgb_image = tf.image.decode_image(image, channels=3)
        rgb_image = tf.image.resize(rgb_image, picture_shape[:-1])        
        rgb_image = rgb_image/255.0        

        return rgb_image 
    


