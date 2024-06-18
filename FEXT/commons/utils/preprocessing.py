import os
import random

from FEXT.commons.configurations import NUM_OF_SAMPLES, TEST_SIZE, VALIDATION_SIZE

#------------------------------------------------------------------------------
def get_images_path(path, num_images=None):

    '''
    Build a dictionary where each key is the file name and the value is the path to images 
    in a given directory.

    Keyword arguments:
        path (str): A string containing the path where the images are located

    Returns:
        dict: A dictionary where keys are image names and values are their corresponding paths
    '''
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    images_dict = {}

    for root, _, files in os.walk(path):
        if num_images is not None:                        
            files = random.sample(files, num_images)
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                images_dict[file] = os.path.join(root, file)

    return images_dict

# [VALIDATION OF DATA]
#------------------------------------------------------------------------------
class DataSplit:

    def __init__(self, images_dictionary):
        
        # get num of samples in train, validation and test datasets
        self.train_size = int(NUM_OF_SAMPLES * (1.0 - TEST_SIZE - VALIDATION_SIZE))
        self.val_size = int(NUM_OF_SAMPLES * VALIDATION_SIZE)
        self.test_size = int(NUM_OF_SAMPLES * TEST_SIZE)
        self.images_dictionary = images_dictionary

    def split_data(self):        
        
        items = list(self.images_dictionary.items())

        # Split the list of items based on the specified sizes
        train_data = dict(items[:self.train_size])
        validation_data = dict(items[self.train_size:self.train_size + self.val_size])
        test_data = dict(items[self.train_size + self.val_size:])       

        return train_data, validation_data, test_data

   