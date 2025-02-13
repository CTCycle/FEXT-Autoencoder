import numpy as np


# [DATA SPLITTING]
###############################################################################
class TrainValidationSplit:

    def __init__(self, images_path : list, configuration):        
        
        self.images_path = images_path
        validation_size = configuration["dataset"]["VALIDATION_SIZE"]
        # shuffle the paths list to perform randomic sampling
        np.random.seed(configuration["dataset"]["SPLIT_SEED"])    
        np.random.shuffle(images_path)            
        # get num of samples in train and validation dataset
        self.train_size = int(len(images_path) * (1.0 - validation_size))
        self.val_size = int(len(images_path) * validation_size)        
        
    #--------------------------------------------------------------------------
    def split_train_and_validation(self): 
        shuffled_indices = np.random.permutation(len(self.images_path))     
        train_indices = shuffled_indices[:self.train_size]
        validation_indices = shuffled_indices[self.train_size:]        
        train_data = [self.images_path[i] for i in train_indices]
        validation_data = [self.images_path[i] for i in validation_indices]
        
        return train_data, validation_data

   