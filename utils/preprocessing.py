import os
import pandas as pd

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



