from typing import List
from FEXT.commons.configurations import TEST_SIZE, VALIDATION_SIZE


# [DATA SPLITTING]
#------------------------------------------------------------------------------
class DataSplit:

    def __init__(self, images_path: List[str]):
        if not isinstance(images_path, list):
            raise TypeError('images_path must be a list')
        
        # get num of samples in train, validation and test datasets
        self.train_size = int(len(images_path) * (1.0 - TEST_SIZE - VALIDATION_SIZE))
        self.val_size = int(len(images_path) * VALIDATION_SIZE)
        self.test_size = int(len(images_path) * TEST_SIZE)
        self.images_path = images_path

    def split_data(self):          
       
        # Split the list of items based on the specified sizes
        train_data = self.images_path[:self.train_size]
        validation_data = self.images_path[self.train_size:self.train_size + self.val_size]
        test_data = self.images_path[self.train_size + self.val_size:]      

        return train_data, validation_data, test_data

   