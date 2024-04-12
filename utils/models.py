import os
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import matplotlib.pyplot as plt



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
    
    def __init__(self, model, device='CPU', seed=42, use_mixed_precision=False, compiled=True):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.model = model
        self.use_mixed_precision = use_mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        self.available_devices = torch.cuda.device_count()
        print(f'\n{self.available_devices} GPU(s) are available.' if self.available_devices else 'Only CPU is available.')
        
        self.set_device(device)

    #--------------------------------------------------------------------------
    def set_device(self, device):
        if device.upper() == 'GPU' and self.available_devices:
            self.device = torch.device('cuda')
            self.model.cuda()
            print('GPU is set as the active device.')
            if self.use_mixed_precision:
                print('Mixed precision training is enabled.')
        else:
            self.device = torch.device('cpu')
            print('CPU is set as the active device.')
        self.model.to(self.device)

    #--------------------------------------------------------------------------
    def train_model(self, data, validation_data, epochs, learning_rate,
                    plot_history=True, plot_frequency=1, plot_path='./'):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        metric_fn = nn.CosineSimilarity(dim=1)

        train_losses, val_losses, train_metrics, val_metrics, epoch_nums = [], [], [], [], []

        for epoch in range(epochs):
            print(f'\nEpoch [{epoch+1}/{epochs}]')
            train_results = self.process_epoch(data, optimizer, loss_fn, metric_fn, train=True)
            val_results = self.process_epoch(validation_data, optimizer, loss_fn, metric_fn, train=False)

            train_losses.append(train_results['loss'])
            train_metrics.append(train_results['metric'])
            val_losses.append(val_results['loss'])
            val_metrics.append(val_results['metric'])
            epoch_nums.append(epoch + 1)

            if plot_history and (epoch + 1) % plot_frequency == 0:
                self.realtime_training_plot((epoch_nums, train_losses, train_metrics, val_losses, val_metrics), plot_frequency, plot_path)

    #--------------------------------------------------------------------------
    def process_epoch(self, data, optimizer, loss_fn, metric_fn, train=True):
        phase = 'train' if train else 'val'
        running_loss, running_metric, total_samples = 0.0, 0.0, 0

        for inputs, targets in tqdm(data, desc=f"Processing {phase} data"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            predictions = self.model(inputs)
            loss = loss_fn(predictions, targets)
            metric = metric_fn(predictions, targets).mean()

            if train:
                optimizer.zero_grad()
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        loss.backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_metric += metric.item() * inputs.size(0)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_metric = running_metric / total_samples
        print(f'{phase.title()} Loss: {epoch_loss:.4f}, Metric: {epoch_metric:.4f}')

        return {'loss': epoch_loss, 'metric': epoch_metric} 
    

    #--------------------------------------------------------------------------
    def realtime_training_plot(self, data, plot_frequency, plot_path):

        epochs, train_losses, train_metrics, val_losses, val_metrics = data        
        if epochs[-1] % plot_frequency == 0:
            fig_path = os.path.join(plot_path, 'training_history.jpeg')
            plt.figure(figsize=(10, 8))
            
            # Plotting training and validation loss
            plt.subplot(2, 1, 1)
            plt.plot(epochs, train_losses, label='Training Loss')
            plt.plot(epochs, val_losses, label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plotting training and validation metrics
            plt.subplot(2, 1, 2)
            plt.plot(epochs, train_metrics, label='Training Metric')
            plt.plot(epochs, val_metrics, label='Validation Metric')
            plt.title('Training and Validation Metric')
            plt.xlabel('Epochs')
            plt.ylabel('Metric')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()

    #--------------------------------------------------------------------------
    def save_model(self, model, path, save_parameters=False, parameters=None):

        model_subfolder = os.path.join(path, 'model')
        os.mkdir(model_subfolder) if not os.path.exists(model_subfolder) else None
        file_path = os.path.join(model_subfolder, 'model_pretrained.pth')        
        torch.save(model.state_dict(), file_path)

        if save_parameters==True and parameters is not None:
            path = os.path.join(path, 'model_parameters.json')      
            with open(path, 'w') as f:
                json.dump(parameters, f)   



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
    
    
    