import os
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary



# function to create a folder where to save model checkpoints
#------------------------------------------------------------------------------
def model_savefolder(path, model_name):

    '''
    Creates a folder with the current date and time to save the model.
    
    Keyword arguments:
        path (str):       A string containing the path where the folder will be created.
        model_name (str): A string containing the name of the model.
    
    Returns:
        str: A string containing the path of the folder where the model will be saved.
        
    '''        
    today_datetime = str(datetime.now())
    truncated_datetime = today_datetime[:-10]
    today_datetime = truncated_datetime.replace(':', '').replace('-', '').replace(' ', 'H') 
    folder_name = f'{model_name}_{today_datetime}'
    model_folder_path = os.path.join(path, folder_name)
    if not os.path.exists(model_folder_path):
        os.mkdir(model_folder_path) 
                    
    return model_folder_path, folder_name


# [POOLING CONVOLUTIONAL BLOCKS]
#==============================================================================
class PooledConvBlock(nn.Module):
    def __init__(self, units, kernel_size, channels, num_layers=2, seed=42, **kwargs):
        super(PooledConvBlock, self).__init__(**kwargs)
        torch.manual_seed(seed)                   
        
        self.activation = nn.ReLU()
        self.pooling = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=0) 
        self.convolutions = nn.ModuleList()
        for i in range(num_layers):     
            current_channels = channels if i == 0 else units      
            self.convolutions.append(nn.Conv2d(in_channels=current_channels,
                                               out_channels=units,
                                               kernel_size=kernel_size,
                                               padding=(2,2)))     

    # implement forward pass
    #--------------------------------------------------------------------------   
    def forward(self, x):
            
        for conv in self.convolutions:
            x = conv(x)
            x = self.activation(x)
        x = self.pooling(x)
        
        return x  
    

# [POOLING TRANSPOSE CONVOLUTIONAL BLOCKS]
#==============================================================================
class TransposeConvBlock(nn.Module):
    def __init__(self, units, kernel_size, channels, num_layers=3, seed=42, **kwargs):
        super(TransposeConvBlock, self).__init__(**kwargs)
        torch.manual_seed(seed) 
        
        self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')       
        self.deconvolutions = nn.ModuleList()
        for i in range(num_layers):  
            current_channels = channels if i == 0 else units      
            self.deconvolutions.append(nn.ConvTranspose2d(in_channels=current_channels, 
                                                          out_channels=units, 
                                                          kernel_size=kernel_size, 
                                                          padding=(1,1)))
        self.activation = nn.ReLU()

    # implement forward pass
    #--------------------------------------------------------------------------
    def forward(self, x):         
        for conv in self.deconvolutions:            
            x = conv(x)
            x = self.activation(x)
        x = self.upsamp(x)       

        return x
    
       
# [MACHINE LEARNING MODELS]
#==============================================================================
class FeXTEncoder(nn.Module):
    def __init__(self, seed=42, **kwargs):
        super(FeXTEncoder, self).__init__(**kwargs)
        torch.manual_seed(seed)
        self.kernel_size = (4,4)

        # Define convolutional blocks based on the previous conversion
        self.convblock1 = PooledConvBlock(64, self.kernel_size, channels=3, num_layers=2, seed=seed)
        self.convblock2 = PooledConvBlock(128, self.kernel_size, channels=64, num_layers=2, seed=seed)
        self.convblock3 = PooledConvBlock(256, self.kernel_size, channels=128, num_layers=2, seed=seed)
        self.convblock4 = PooledConvBlock(512, self.kernel_size, channels=256, num_layers=3, seed=seed)
        self.convblock5 = PooledConvBlock(512, self.kernel_size, channels=512, num_layers=3, seed=seed)
        
    # implement forward pass
    #--------------------------------------------------------------------------
    def forward(self, x):
        # Apply each convolutional block in sequence
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)       

        return x
    

# [MACHINE LEARNING MODELS]
#==============================================================================
class FeXTDecoder(nn.Module):
    def __init__(self, seed=42, **kwargs):
        super(FeXTDecoder, self).__init__(**kwargs)
        torch.manual_seed(seed)
        self.kernel_size = (3,3)
        
        # Initialize transposed convolution blocks
        self.convblock1 = TransposeConvBlock(512, self.kernel_size, channels=512, num_layers=3, seed=seed)
        self.convblock2 = TransposeConvBlock(512, self.kernel_size, channels=512, num_layers=3, seed=seed)
        self.convblock3 = TransposeConvBlock(256, self.kernel_size, channels=512, num_layers=2, seed=seed)
        self.convblock4 = TransposeConvBlock(128, self.kernel_size, channels=256, num_layers=2, seed=seed)
        self.convblock5 = TransposeConvBlock(3, self.kernel_size, channels=128, num_layers=2, seed=seed)     
        self.activation = nn.Sigmoid()

    # implement forward pass
    #--------------------------------------------------------------------------
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)   
        x = self.activation(x)

        return x     


# [MACHINE LEARNING MODELS]
#==============================================================================
class FeXTAutoEncoder(nn.Module):
    def __init__(self, seed=42, **kwargs):
        super(FeXTAutoEncoder, self).__init__(**kwargs)
        self.encoder = FeXTEncoder(seed)
        self.decoder = FeXTDecoder(seed)               

    # implement forward pass
    #--------------------------------------------------------------------------
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)                                                

        return x
    
    #--------------------------------------------------------------------------
    def print_summary(self):
        summary(self, input_size=(1, 3, 256, 256))
       

# [TRAINING MACHINE LEARNING MODELS]
#==============================================================================
class ModelTraining:
    
    def __init__(self, device='CPU', seed=42, use_mixed_precision=False, compiled=True):                            
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.use_mixed_precision = use_mixed_precision
        self.compiled = compiled
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  
        
        self.available_devices = torch.cuda.device_count()        
        print(f'\n{self.available_devices} GPU(s) are available.' if self.available_devices else 'Only CPU is available.') 
        
        if device.upper() == 'GPU' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('GPU is set as the active device.')
            if use_mixed_precision:
                print('Mixed precision training is enabled.')
        else:
            self.device = torch.device('cpu')
            print('CPU is set as the active device.')        
    
    #--------------------------------------------------------------------------
    def get_device(self):
        return self.device

    #--------------------------------------------------------------------------
    def get_scaler(self):
        if self.use_mixed_precision:
            return self.scaler
        else:
            return None   

    #--------------------------------------------------------------------------
    def train_model(self, model, data, validation_data, epochs, learning_rate):
    
        # compile model if requested
        if self.compiled:
            model = torch.compile(model)
        
        # load the model onto device
        device = self.get_device()
        model.to(device) 

        # Optimizer and Loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()        
        metric_fn = nn.CosineSimilarity(dim=1)         
        
        # loop over epochs
        for i, epoch in enumerate(range(epochs)):
            model.train()  # Set the model to training mode
            training_loss = 0.0
            training_metric = 0.0
            validation_loss = 0.0
            validation_metric = 0.0

            print(f'\nEpoch [{epoch+1}/{epochs}]') 
            for inputs, targets in tqdm(data):
                inputs, targets = inputs.to(device), targets.to(device)

                if self.use_mixed_precision==True:
                    # Mixed precision training block
                    with torch.cuda.amp.autocast(enabled=True):
                        predictions = model(inputs)                               
                        loss = loss_fn(predictions, targets)
                        metric = metric_fn(predictions, targets).mean()
                    optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # Float32 precision training block
                    predictions = model(inputs)                               
                    loss = loss_fn(predictions, targets)
                    metric = metric_fn(predictions, targets).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()                
                
                training_loss += loss.item() * inputs.size(0)
                training_metric += metric.item() * inputs.size(0)
            
            # Validation phase
            model.eval()  # Set the model to evaluation mode            
            with torch.no_grad():  # Disable gradient computation during validation
                for inputs, targets in validation_data:
                    inputs, targets = inputs.to(device), targets.to(device)
                    predictions = model(inputs)
                    loss = loss_fn(predictions, targets)
                    metric = metric_fn(predictions, targets).mean()  # Calculate mean metric for batch                    
                    validation_loss += loss.item() * inputs.size(0)
                    validation_metric += metric.item() * inputs.size(0)

            # Calculate average losses and metrics
            epoch_training_loss = training_loss / len(data.dataset)
            epoch_training_metric = training_metric / len(data.dataset)
            epoch_validation_loss = validation_loss / len(validation_data.dataset)
            epoch_validation_metric = validation_metric / len(validation_data.dataset)
            
            # Print epoch-level statistics
            print(f'\nTrain data - Loss: {epoch_training_loss:.4f}, Metric: {epoch_training_metric:.4f}')
            print(f'Validation data - Loss: {epoch_validation_loss:.4f}, Metric: {epoch_validation_metric:.4f}')

    #--------------------------------------------------------------------------
    def save_model(self, model, path):

        file_path = os.path.join(path, 'model_pretrained.pth')        
        torch.save(model.state_dict(), file_path)


# [SAVE MODEL PARAMS]       
#------------------------------------------------------------------------------ 
def model_parameters(parameters_dict, savepath):

    '''
    Saves the model parameters to a JSON file. The parameters are provided 
    as a dictionary and are written to a file named 'model_parameters.json' 
    in the specified directory.

    Keyword arguments:
        parameters_dict (dict): A dictionary containing the parameters to be saved.
        savepath (str): The directory path where the parameters will be saved.

    Returns:
        None       

    '''
    path = os.path.join(savepath, 'model_parameters.json')      
    with open(path, 'w') as f:
        json.dump(parameters_dict, f) 


# [INFERENCE]
#==============================================================================
class Inference:

    def __init__(self, seed):
        self.seed = seed
        torch.manual_seed(seed)

    #--------------------------------------------------------------------------
    def load_pretrained_model(self, path):

        '''
        Load a pretrained PyTorch model (state dict) from the specified directory. 
        If multiple model directories are found, the user is prompted to select one,
        while if only one model directory is found, that model is loaded directly.
        If `load_parameters` is True, the function also loads the model parameters 
        from the target .pth file in the same directory.

        Keyword arguments:
            path (str): The directory path where the pretrained models are stored.            

        Returns:
            model (torch.nn.Module): The loaded PyTorch model.
            configuration (dict, optional): The loaded model parameters, if available.
            
        '''
        model_folders = [entry.name for entry in os.scandir(path) if entry.is_dir()]
        
        if len(model_folders) > 1:
            model_folders.sort()
            print('Please select a pretrained model:') 
            for i, directory in enumerate(model_folders):
                print(f'{i + 1} - {directory}')
            print()
            dir_index = int(input('Type the model index to select it: '))
            while dir_index not in range(1, len(model_folders) + 1):
                dir_index = int(input('Input is not valid! Try again: '))
            folder_path = os.path.join(path, model_folders[dir_index - 1])
        elif len(model_folders) == 1:
            folder_path = os.path.join(path, model_folders[0])
        else:
            raise FileNotFoundError('No model directories found.')

        model = FeXTAutoEncoder(seed=42)  # Initialize your PyTorch model class here        
        model_path = os.path.join(folder_path, 'model', 'pretrained_model.pth')
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
        
        # Load configuration if exists
        config_path = os.path.join(folder_path, 'model_parameters.json')
        configuration = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                configuration = json.load(f)
        
        return model, configuration
    
    #--------------------------------------------------------------------------
    def model_inference(self, data, model):        
        
        # load the model onto device
        device = self.get_device()
        model.to(device) 
        predictions = []         
        for inputs, targets in tqdm(data):
            inputs, targets = inputs.to(device), targets.to(device)
            prediction = model(inputs)
            predictions.append(prediction)

        return predictions

        
#------------------------------------------------------------------------------
if __name__ == '__main__':  
        
    print('PyTorch Version:', torch.__version__)
    
    
    