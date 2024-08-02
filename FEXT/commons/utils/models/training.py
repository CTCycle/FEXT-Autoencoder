import os
import torch
import numpy as np
import tensorflow as tf
import keras

from FEXT.commons.utils.models.callbacks import RealTimeHistory, LoggingCallback
from FEXT.commons.utils.dataloader.serializer import ModelSerializer
from FEXT.commons.constants import CONFIG
from FEXT.commons.logger import logger


# [TOOLS FOR TRAINING MACHINE LEARNING MODELS]
###############################################################################
class ModelTraining:    
       
    def __init__(self):
        np.random.seed(CONFIG["SEED"])
        self.device = torch.device('cpu')
        self.set_device()
        if self.device.type=='cuda' and CONFIG["training"]["MIXED_PRECISION"]:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None  
       

    # set device
    #--------------------------------------------------------------------------
    def set_device(self):

        if CONFIG["training"]["ML_DEVICE"] == 'GPU':
            if not torch.cuda.is_available():
                logger.info('No GPU found. Falling back to CPU')
                self.device = torch.device('cpu')
            else:
                self.device = torch.device('cuda:0')
                if CONFIG["training"]["MIXED_PRECISION"]:
                    logger.info('Mixed precision policy is active during training')
                torch.cuda.set_device(self.device)
                logger.info('GPU is set as active device')
        elif CONFIG["training"]["ML_DEVICE"] == 'CPU':
            self.device = torch.device('cpu')
            logger.info('CPU is set as active device')
        else:
            logger.error(f'Unknown ML_DEVICE value: {CONFIG["training"]["ML_DEVICE"]}')            
            self.device = torch.device('cpu')

    #--------------------------------------------------------------------------
    def train_model(self, model : keras.Model, train_data, 
                    validation_data, current_checkpoint_path, from_epoch=0,
                    session_index=0):

        # initialize the real time history callback    
        RTH_callback = RealTimeHistory(current_checkpoint_path)
        logger_callback = LoggingCallback()
        callbacks_list = [RTH_callback, logger_callback]

        # initialize tensorboard if requested    
        if CONFIG["training"]["USE_TENSORBOARD"]:
            logger.debug('Using tensorboard during training')
            log_path = os.path.join(current_checkpoint_path, 'tensorboard')
            callbacks_list.append(keras.callbacks.TensorBoard(log_dir=log_path, 
                                                                 histogram_freq=1))

        # training loop and save model at end of training
        serializer = ModelSerializer()         

        # calculate number of epochs taking into account possible training resumption
        additional_epochs = from_epoch if session_index > 0 else 0
        epochs = CONFIG["training"]["EPOCHS"] + additional_epochs 
        
        training = model.fit(train_data, epochs=epochs, validation_data=validation_data, 
                            callbacks=callbacks_list, initial_epoch=from_epoch)

        serializer.save_pretrained_model(model, current_checkpoint_path)

        # save model parameters in json files         
        parameters = {'picture_shape' : CONFIG["model"]["IMG_SHAPE"],                           
                      'augmentation' : CONFIG["dataset"]["IMG_AUGMENT"],              
                      'batch_size' : CONFIG["training"]["BATCH_SIZE"],
                      'learning_rate' : CONFIG["training"]["LEARNING_RATE"],
                      'epochs' : epochs,
                      'seed' : CONFIG["SEED"],
                      'tensorboard' : CONFIG["training"]["USE_TENSORBOARD"],
                      'session_ID': session_index}

        serializer.save_model_parameters(current_checkpoint_path, parameters)

        


      