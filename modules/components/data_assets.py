import os
from datetime import datetime
import tensorflow as tf
from tqdm import tqdm

# [CONSOLE USER OPERATIONS]
#==============================================================================
# Series of methods to interact with the user through console
#==============================================================================
class UserOperations:   
    
    
    #--------------------------------------------------------------------------
    def menu_selection(self, menu):
        
        '''         
        Presents a menu to the user and returns the selected option.
        
        Keyword arguments:                      
            menu (dict): A dictionary containing the options to be presented to the user. 
                         The keys are integers representing the option numbers, and the 
                         values are strings representing the option descriptions.
        
        Returns:            
            op_sel (int): The selected option number.
        
        '''      
        indexes = [idx + 1 for idx, val in enumerate(menu)]
        for key, value in menu.items():
            print(f'{key} - {value}')
        print()        
        while True:
            try:
                op_sel = int(input('Select the desired operation: '))
            except:
                continue             
            while op_sel not in indexes:
                try:
                    op_sel = int(input('Input is not valid, please select a valid option: '))
                except:
                    continue
            break
        
        return op_sel                
            

# [PREPROCESSING PIPELINE]
#==============================================================================
# Series of methods and functions to preprocess data for model training
#==============================================================================
class PreProcessing:    

    #--------------------------------------------------------------------------
    def images_pathfinder(self, path, dataframe, id_col):

        '''
        Add a column with relative path to images in a dataframe, given a column where the
        images names are stored
    
        Keyword arguments:
            path (str):         A string containing the path where the images are located
            dataframe (pandas): the selected dataframe
            id_col (str):       Name of the column with images names 
    
        Returns:
            dataframe: the modified dataframe
        
        '''
        images_paths = {}
        for pic in os.listdir(path):
            pic_name = pic.split('.')[0]
            pic_path = os.path.join(path, pic)                        
            path_pair = {pic_name : pic_path}        
            images_paths.update(path_pair)       
        dataframe['images_path'] = dataframe[id_col].map(images_paths)
        dataframe = dataframe.dropna(subset=['images_path']).reset_index(drop = True)

        return dataframe

    #--------------------------------------------------------------------------
    def load_images(self, paths, image_size):
        
        images = []
        for pt in tqdm(paths):
            image = tf.io.read_file(pt)
            image = tf.image.decode_image(image, channels=3)
            image = tf.image.resize(image, image_size)
            image = tf.reverse(image, axis=[-1])
            image = image/255.0 
            images.append(image) 

        return images    
    
    #--------------------------------------------------------------------------
    def model_savefolder(self, path, model_name):

        '''
        Creates a folder with the current date and time to save the model.
    
        Keyword arguments:
            path (str):       A string containing the path where the folder will be created.
            model_name (str): A string containing the name of the model.
    
        Returns:
            str: A string containing the path of the folder where the model will be saved.
        
        '''        
        today_datetime = str(datetime.now())
        truncated_datetime = today_datetime[:-10]
        today_datetime = truncated_datetime.replace(':', '').replace('-', '').replace(' ', 'H') 
        self.folder_name = f'{model_name}_{today_datetime}'
        model_folder_path = os.path.join(path, self.folder_name)
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path) 
                    
        return model_folder_path   


    
    
    
        
      
