import keras
import torch

from FEXT.commons.utils.learning.callbacks import initialize_callbacks_handler
from FEXT.commons.utils.data.serializer import ModelSerializer
from FEXT.commons.constants import CONFIG
from FEXT.commons.logger import logger


# [TOOLS FOR TRAINING MACHINE LEARNING MODELS]
###############################################################################
class ModelTraining:    
       
    def __init__(self, configuration : dict):
        self.serializer = ModelSerializer()        
        keras.utils.set_random_seed(configuration.get('training_seed', 42))        
        self.selected_device = configuration.get('device', 'CPU')
        self.device_id = configuration.get('device_ID', 0)
        self.mixed_precision = configuration.get('use_mixed_precision', False)        
        self.configuration = configuration     

    # set device
    #--------------------------------------------------------------------------
    def set_device(self):
        if self.selected_device == 'GPU':
            if not torch.cuda.is_available():
                logger.info('No GPU found. Falling back to CPU')
                self.device = torch.device('cpu')
            else:
                self.device = torch.device(f'cuda:{self.device_id}')
                torch.cuda.set_device(self.device)  
                logger.info('GPU is set as active device')            
                if self.mixed_precision:
                    keras.mixed_precision.set_global_policy("mixed_float16")
                    logger.info('Mixed precision policy is active during training')                   
        else:
            self.device = torch.device('cpu')
            logger.info('CPU is set as active device')

    #--------------------------------------------------------------------------
    def get_training_history(self, train_history, val_history, epochs):   
        # use the real time history callback data to retrieve current loss and metric values
        # this allows to correctly resume the training metrics plot if training from checkpoint        
        session = {'history' : train_history, 
                   'val_history' : val_history,
                   'total_epochs' : epochs}
        
        return session
        
    #--------------------------------------------------------------------------
    def train_model(self, model : keras.Model, train_data, validation_data, 
                    checkpoint_path, from_checkpoint=False, progress_callback=None):         

        # perform different initialization duties based on state of session:
        # training from scratch vs resumed training
        # calculate number of epochs taking into account possible training resumption
        if not from_checkpoint:            
            epochs = self.configuration.get('num_epochs', 10) 
            from_epoch = 0
            history = None
        else:
            _, history = self.serializer.load_session_configuration(checkpoint_path)                     
            epochs = history['total_epochs'] + self.configuration.get('additional_epochs', 10) 
            from_epoch = history['total_epochs']           
       
        # add all callbacks to the callback list
        callbacks_list = initialize_callbacks_handler(
            self.configuration, checkpoint_path, history, progress_callback)       
        
        # run model fit using keras API method.             
        training = model.fit(train_data, epochs=epochs, validation_data=validation_data, 
                             callbacks=callbacks_list, initial_epoch=from_epoch)

        self.get_training_history(None, None, epochs)           
        self.serializer.save_pretrained_model(model, checkpoint_path)       
        self.serializer.save_session_configuration(
            checkpoint_path, history, self.configuration)

        


      