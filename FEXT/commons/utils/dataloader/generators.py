import numpy as np
import tensorflow as tf
import torchvision.transforms as T
import pandas as pd

import torch
from torch.utils.data import Dataset
from PIL import Image

from FEXT.commons.constants import CONFIG
from FEXT.commons.logger import logger

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from FEXT.commons.constants import CONFIG
from FEXT.commons.logger import logger


###############################################################################
# [CUSTOM DATA GENERATOR for PyTorch]
###############################################################################
class DataGenerator:
    
    def __init__(self, configuration):
        
        self.img_shape = configuration["model"]["IMG_SHAPE"] 
        self.augmentation = configuration["dataset"]["IMG_AUGMENT"]
        self.configuration = configuration

        # Build the transformation pipeline:
        #  - Always resize to (H, W).
        #  - Perform optional augmentation.
        #  - Convert to tensor [0,1].
        self.transform = self._build_transform_pipeline()

    #--------------------------------------------------------------------------
    def _build_transform_pipeline(self):
        
        H, W, _ = self.img_shape
        transform_list = []

        # Data augmentation transforms (applied if self.augmentation == True):
        if self.augmentation:
            # Flip horizontally with p=0.5
            transform_list.append(T.RandomHorizontalFlip(p=0.5))
            # Flip vertically with p=0.5
            transform_list.append(T.RandomVerticalFlip(p=0.5))

            # Brightness with probability 0.25
            #   max_delta=0.2 => brightness factor in [0.8, 1.2].
            transform_list.append(T.RandomApply([T.ColorJitter(brightness=0.2)], p=0.25))
            
            # Contrast with probability 0.35
            #   lower=0.7, upper=1.3 => factor in [0.7, 1.3].
            #   This equates to contrast=0.3 in ColorJitter, but we isolate it:
            transform_list.append(T.RandomApply([T.ColorJitter(contrast=0.3)], p=0.35))

        # Always resize to the given shape
        transform_list.append(T.Resize((H, W)))

        # Finally convert the image to a tensor in [0,1]
        transform_list.append(T.ToTensor())

        return T.Compose(transform_list)

    #--------------------------------------------------------------------------
    def get_data(self, path):
        
        # Load image with PIL
        with Image.open(path).convert("RGB") as img:
            image_tensor = self.transform(img)

            # Permute to channels-last format (H, W, C)
            image_tensor = image_tensor.permute(1, 2, 0)

        # Return (input, label) as in the original code
        return image_tensor, image_tensor


###############################################################################
# [CUSTOM DATASET]
###############################################################################
class TensorDataset(Dataset):
    
    def __init__(self, data : list, configuration):        
        self.data = data  
        self.generator = DataGenerator(configuration)

    #--------------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    #--------------------------------------------------------------------------
    def __getitem__(self, idx):           
        path = self.data[idx] 
        image_tensor, label_tensor = self.generator.get_data(path)

        return image_tensor, label_tensor


###############################################################################
# [DATA LOADER BUILDER]
###############################################################################
class DataLoaderBuilder:

    def __init__(self, configuration):
        self.num_workers = 6       
        self.configuration = configuration

    #--------------------------------------------------------------------------
    def build_model_dataloader(self, train_data: pd.DataFrame, validation_data: pd.DataFrame,
                               batch_size = None):
        

        # Fallback to config batch_size if none provided
        batch_size = self.configuration["training"]["BATCH_SIZE"] if batch_size is None else batch_size        
       
        train_dataset = TensorDataset(train_data, self.configuration)
        val_dataset = TensorDataset(validation_data, self.configuration)
       
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                  shuffle=True, num_workers=self.num_workers)

        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader

        
