import os
import random

from FEXT.commons.configurations import NUM_OF_SAMPLES, TEST_SIZE, VALIDATION_SIZE
from FEXT.commons.pathfinder import IMG_DATA_PATH

#------------------------------------------------------------------------------
def get_images_path():

    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    images_path = []

    for root, _, files in os.walk(IMG_DATA_PATH):
        if NUM_OF_SAMPLES is not None:                        
            files = random.sample(files, NUM_OF_SAMPLES)
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                images_path.append(os.path.join(root, file))                

    return images_path


# [VALIDATION OF DATA]
#------------------------------------------------------------------------------
class DataSplit:

    def __init__(self, images_path):
        
        # get num of samples in train, validation and test datasets
        self.train_size = int(NUM_OF_SAMPLES * (1.0 - TEST_SIZE - VALIDATION_SIZE))
        self.val_size = int(NUM_OF_SAMPLES * VALIDATION_SIZE)
        self.test_size = int(NUM_OF_SAMPLES * TEST_SIZE)
        self.images_path = images_path

    def split_data(self):          
       
        # Split the list of items based on the specified sizes
        train_data = self.images_path[:self.train_size]
        validation_data = self.images_path[self.train_size:self.train_size + self.val_size]
        test_data = self.images_path[self.train_size + self.val_size:]      

        return train_data, validation_data, test_data

   