import torch  
import cv2 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# [CUSTOM DATA GENERATOR FOR TRAINING]
#==============================================================================
class DataGenerator(Dataset):

    def __init__(self, dataframe, picture_shape=(256, 256, 3), shuffle=True, 
                 augmentation=True, normalization=True):
        self.dataframe = dataframe
        self.path_col = 'path'
        self.picture_shape = picture_shape
        self.augmentation = augmentation
        self.normalization = normalization
        self.shuffle = shuffle
        
        if self.shuffle:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

        # Define transformations
        self.transformations = []
        if self.augmentation:
            self.transformations.extend([transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip()])
        if self.normalization:
            self.transformations.append(transforms.ToTensor())
            
        self.transform = transforms.Compose(self.transformations)

    # define length of the custom generator      
    #--------------------------------------------------------------------------
    def __len__(self):
        return len(self.dataframe)

    # define method to get X and Y data through custom functions, and subsequently
    # create a batch of data converted to tensors
    #--------------------------------------------------------------------------
    def __getitem__(self, idx):

        img_path = self.dataframe.iloc[idx][self.path_col]  
        image = cv2.imread(img_path)        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
        image = cv2.resize(image, self.picture_shape[:-1])
        
        if self.transform:
            image = self.transform(image)
        
        return image, image     
     
    
# [CREATE DATA LOADER]    
#------------------------------------------------------------------------------
def dataloader(dataframe, batch_size, picture_shape, shuffle=True, 
               augmentation=True, normalization=True, device='CPU', num_workers=0):
    dataset = DataGenerator(dataframe, picture_shape, shuffle, augmentation, normalization)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            pin_memory=True if device == 'GPU' else False)

    return dataloader    
    
           
#------------------------------------------------------------------------------
if __name__ == '__main__':  
        
    print('PyTorch Version:', torch.__version__)
    
    if torch.cuda.is_available():
        print('CUDA is available. GPU support is enabled.')
        print('Number of GPUs available:', torch.cuda.device_count())
        print('Name of the GPU:', torch.cuda.get_device_name(0))
    else:
        print('CUDA is not available. GPU support is not enabled.')   
            
