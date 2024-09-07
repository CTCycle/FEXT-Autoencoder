import numpy as np
from FEXT.commons.constants import CONFIG


# [DATA SPLITTING]
###############################################################################
class DataSplit:

    def __init__(self, images_path : list):        
        
        self.images_path = images_path
        validation_size = CONFIG["dataset"]["VALIDATION_SIZE"]

        # shuffle the paths list to perform randomic sampling
        np.random.seed(CONFIG["dataset"]["SPLIT_SEED"])    
        np.random.shuffle(images_path)    
        
        # get num of samples in train and validation dataset
        self.train_size = int(len(images_path) * (1.0 - validation_size))
        self.val_size = int(len(images_path) * validation_size)        
        
    #--------------------------------------------------------------------------
    def split_train_and_validation(self):          
       
        # Split the list of items based on the specified sizes
        train_data = self.images_path[:self.train_size]
        validation_data = self.images_path[self.train_size:]
        
        return train_data, validation_data

   