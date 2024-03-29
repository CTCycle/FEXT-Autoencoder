import os
import cv2
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import tensorflow as tf



# function to create a folder where to save model checkpoints
#------------------------------------------------------------------------------
def model_savefolder(path, model_name):

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
        folder_name = f'{model_name}_{today_datetime}'
        model_folder_path = os.path.join(path, folder_name)
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path) 
                    
        return model_folder_path, folder_name


#------------------------------------------------------------------------------
def dataset_from_images(path, dataset=None):

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
        if dataset is None:
            image_locations = []
            image_names = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    image_locations.append(os.path.join(root, file))
                    image_names.append(file)            
            dataset = pd.DataFrame({'name': image_names, 'path': image_locations})  
        else:      
            dataset['path'] = dataset['name'].apply(lambda x : os.path.join(path, x))  

        return dataset

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

