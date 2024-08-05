import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import tensorflow as tf
import keras

from FEXT.commons.constants import CONFIG
from FEXT.commons.logger import logger
             
        
# [CUSTOM DATA GENERATOR FOR TRAINING]
###############################################################################
# Generate and preprocess input and output for the machine learning model and build
# a tensor dataset with prefetching and batching
###############################################################################
class DataGenerator(Dataset):

    def __init__(self, data):              
        
        self.data = data
        self.img_shape = CONFIG["model"]["IMG_SHAPE"]       
        self.normalization = CONFIG["dataset"]["IMG_NORMALIZE"]
        self.augmentation = CONFIG["dataset"]["IMG_AUGMENT"]        
        self.transform = transforms.Compose(self.get_transforms()) 
        
    
    # load and preprocess a single image
    #--------------------------------------------------------------------------
    def load_image(self, path):

        '''
        Loads and preprocesses a single image.

        Keyword arguments:
            path (str): The path to the image file.

        Returns:
            rgb_image (tf.Tensor): The preprocessed RGB image tensor.

        '''
        image = Image.open(path).convert('RGB')        
        image = self.transform(image)

        return image   
    
    # define method perform data augmentation    
    #--------------------------------------------------------------------------
    def image_normalization(self, image):

        image = image/255.0

        return image

    # define method perform data augmentation    
    #--------------------------------------------------------------------------
    def get_transforms(self):

        transform_list = [transforms.Resize(self.img_shape[:-1])]
        
        if self.augmentation:
            transform_list.extend([transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.RandomAffine(degrees=0, translate=(0.2, 0.3))])
                   
        transform_list.extend([transforms.ToTensor(),
                              transforms.Lambda(self.image_normalization)])           
               
        return transform_list
    
    # define method perform data augmentation    
    #--------------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    #--------------------------------------------------------------------------
    def __getitem__(self, idx):
        
        '''
        Loads and preprocesses a single image.
        
        Keyword arguments:
            idx (int): Index of the image to be loaded.
        
        Returns:
            image (torch.Tensor): Preprocessed image tensor.
            target (torch.Tensor): Same as image tensor (as target).
        '''
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return image, image       
   

    
# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
def build_dataloader(train_data, validation_data):    
        
    train_dataset = DataGenerator(train_data)
    val_dataset = DataGenerator(validation_data)
    
    train_gen = DataLoader(train_dataset, batch_size=CONFIG["training"]["BATCH_SIZE"], 
                            shuffle=True, num_workers=CONFIG["training"]["NUM_PROCESSORS"],
                            pin_memory=True)
    
    validation_gen = DataLoader(val_dataset, batch_size=CONFIG["training"]["BATCH_SIZE"], 
                                shuffle=True, num_workers=CONFIG["training"]["NUM_PROCESSORS"],
                                pin_memory=True)
    
    return train_gen, validation_gen


            
