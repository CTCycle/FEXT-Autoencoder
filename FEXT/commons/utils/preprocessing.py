from typing import List
from FEXT.commons.constants import CONFIG


# [DATA SPLITTING]
#------------------------------------------------------------------------------
class DataSplit:

    def __init__(self, images_path: List[str]):
        if not isinstance(images_path, list):
            raise TypeError('images_path must be a list')
        
        validation_size = CONFIG["dataset"]["VALIDATION_SIZE"]
        test_size = CONFIG["dataset"]["TEST_SIZE"]
        
        # get num of samples in train, validation and test datasets
        self.train_size = int(len(images_path) * (1.0 - test_size - validation_size))
        self.val_size = int(len(images_path) * validation_size)
        self.test_size = int(len(images_path) * test_size)
        self.images_path = images_path

    def split_data(self):          
       
        # Split the list of items based on the specified sizes
        train_data = self.images_path[:self.train_size]
        validation_data = self.images_path[self.train_size:self.train_size + self.val_size]
        test_data = self.images_path[self.train_size + self.val_size:]      

        return train_data, validation_data, test_data

   