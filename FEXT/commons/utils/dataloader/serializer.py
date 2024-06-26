import os
import cv2
import json
from tqdm import tqdm
from datetime import datetime
import tensorflow as tf
from keras.utils import plot_model

from FEXT.commons.constants import CONFIG, CHECKPOINT_PATH

    
#------------------------------------------------------------------------------
def get_images_path(path, sample_size=None):

    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    images_path = []

    for root, _, files in os.walk(path):
        if sample_size is not None:
            files = files[:int(sample_size*len(files))]           
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                images_path.append(os.path.join(root, file))                

    return images_path


#------------------------------------------------------------------------------
class DataSerializer:

    def __init__(self):
        
        self.model_name = 'FeXT'
       
    #------------------------------------------------------------------------------
    def load_images(self, paths, as_tensor=True):

        img_shape = CONFIG["model"]["IMG_SHAPE"]           
        images = []
        for pt in tqdm(paths):
            if as_tensor==False:                
                image = cv2.imread(pt)             
                image = cv2.resize(image, img_shape[:-1])            
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                if CONFIG["dataset"]["IMG_NORMALIZE"]:
                    image = image/255.0
            else:
                image = tf.io.read_file(pt)
                image = tf.image.decode_image(image, channels=3)
                image = tf.image.resize(image, img_shape[:-1])
                image = tf.reverse(image, axis=[-1])
                if CONFIG["dataset"]["IMG_NORMALIZE"]:
                    image = image/255.0
            
            images.append(image) 

        return images

    # ...
    #--------------------------------------------------------------------------
    def save_preprocessed_data(self, train_data, validation_data, path):        

        combined_data = {'train': train_data, 
                         'validation': validation_data}

        with open(os.path.join(path, 'preprocessed_data.json'), 'w') as json_file:
            json.dump(combined_data, json_file)

    # ...
    #--------------------------------------------------------------------------
    def load_preprocessed_data(self, path):

        json_file_path = os.path.join(path, 'preprocessed_data.json')    
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"The file {json_file_path} does not exist.")
        
        with open(json_file_path, 'r') as json_file:
            combined_data = json.load(json_file)
        
        train_data = combined_data.get('train')
        validation_data = combined_data.get('validation')        
        
        return {'train': train_data, 
                'validation': validation_data}
        
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
        os.makedirs(checkpoint_folder_path, exist_ok=True)
        self.preprocessing_path = os.path.join(checkpoint_folder_path, 'preprocessing')
        os.makedirs(self.preprocessing_path, exist_ok=True)
        
        return checkpoint_folder_path    
    

# [...]
#------------------------------------------------------------------------------
class ModelSerializer:

    def __init__(self):
        pass

    #--------------------------------------------------------------------------
    def save_pretrained_model(self, model, path):

        model_files_path = os.path.join(path, 'model')
        model.save(model_files_path, save_format='tf')
        print(f'\nTraining session is over. Model has been saved in folder {path}')

    #--------------------------------------------------------------------------
    def save_model_parameters(self, path, parameters_dict):

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
        path = os.path.join(path, 'model_parameters.json')      
        with open(path, 'w') as f:
            json.dump(parameters_dict, f)

    #--------------------------------------------------------------------------
    def save_model_plot(self, model, path):

        if CONFIG["model"]["SAVE_MODEL_PLOT"]:
            plot_path = os.path.join(path, 'model_layout.png')       
            plot_model(model, to_file=plot_path, show_shapes=True, 
                       show_layer_names=True, show_layer_activations=True, 
                       expand_nested=True, rankdir='TB', dpi=400)
            
    #-------------------------------------------------------------------------- 
    def load_pretrained_model(self):

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
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():
                model_folders.append(entry.name)
    
        if not model_folders:
            raise FileNotFoundError('No model directories found in the specified path.')
        
        if len(model_folders) > 1:
            model_folders.sort()
            index_list = [idx + 1 for idx, item in enumerate(model_folders)]     
            print('Currently available pretrained models:') 
            print()
            for i, directory in enumerate(model_folders):
                print(f'{i + 1} - {directory}')        
            print()               
            while True:
                try:
                    dir_index = int(input('Select the pretrained model: '))
                    print()
                except ValueError:
                    continue
                if dir_index in index_list:
                    break
                else:
                    print('Input is not valid! Try again:')
                    
            self.loaded_model_folder = os.path.join(CHECKPOINT_PATH, model_folders[dir_index - 1])

        elif len(model_folders) == 1:
            self.loaded_model_folder = os.path.join(CHECKPOINT_PATH, model_folders[0])                 
            
        model_path = os.path.join(self.loaded_model_folder, 'model') 
        model = tf.keras.models.load_model(model_path)
        
        configuration = None
        config_path = os.path.join(self.loaded_model_folder, 'model_parameters.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                configuration = json.load(f)       
        else:
            print('Warning: model_parameters.json file not found. Model parameters were not loaded.')
            
        return model, configuration