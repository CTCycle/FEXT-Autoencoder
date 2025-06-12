import keras
import torch

from FEXT.commons.utils.learning.callbacks import initialize_callbacks_handler
from FEXT.commons.utils.data.serializer import ModelSerializer
from FEXT.commons.logger import logger


# [TOOLS FOR TRAINING MACHINE LEARNING MODELS]
###############################################################################
class ModelTraining:    
       
    def __init__(self, configuration):
        self.serializer = ModelSerializer()        
        keras.utils.set_random_seed(configuration.get('training_seed', 42))        
        self.selected_device = configuration.get('device', 'CPU')
        self.device_id = configuration.get('device_ID', 0)
        self.mixed_precision = configuration.get('use_mixed_precision', False)        
        self.configuration = configuration     

    # set device
    #--------------------------------------------------------------------------
    def set_device(self, device_override=None):
        selected_device = device_override if device_override else self.selected_device
        if selected_device == 'GPU':
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
    def train_model(self, model, train_data, validation_data, 
                    checkpoint_path, progress_callback=None, worker=None): 
                
        epochs = self.configuration.get('epochs', 10)      
        # add all callbacks to the callback list
        callbacks_list = initialize_callbacks_handler(
            self.configuration, checkpoint_path, 
            progress_callback=progress_callback, worker=worker)       
        
        # run model fit using keras API method.             
        session = model.fit(
            train_data, epochs=epochs, validation_data=validation_data, 
            callbacks=callbacks_list)
                   
        self.serializer.save_pretrained_model(model, checkpoint_path)       
        self.serializer.save_training_configuration(
            checkpoint_path, session, self.configuration)
        
    #--------------------------------------------------------------------------
    def resume_training(self, model, train_data, validation_data, 
                        checkpoint_path, session=None, progress_callback=None,
                        worker=None):  
        
        from_epoch = 0 if not session else session['epochs']     
        total_epochs = from_epoch + self.configuration.get('additional_epochs', 10)           
        # add all callbacks to the callback list
        callbacks_list = initialize_callbacks_handler(
            self.configuration, checkpoint_path, session, progress_callback, worker)       
        
        # run model fit using keras API method.             
        session = model.fit(
            train_data, epochs=total_epochs, validation_data=validation_data, 
            callbacks=callbacks_list, initial_epoch=from_epoch)
                   
        self.serializer.save_pretrained_model(model, checkpoint_path)       
        self.serializer.save_training_configuration(
            checkpoint_path, session, self.configuration)

        


      