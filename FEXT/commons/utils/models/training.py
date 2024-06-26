import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from FEXT.commons.utils.models.callbacks import RealTimeHistory
from FEXT.commons.utils.dataloader.serializer import ModelSerializer
from FEXT.commons.constants import CONFIG


# [TOOLS FOR TRAINING MACHINE LEARNING MODELS]
#------------------------------------------------------------------------------
class ModelTraining:    
       
    def __init__(self):                            
        np.random.seed(CONFIG["SEED"])
        tf.random.set_seed(CONFIG["SEED"])         
        self.available_devices = tf.config.list_physical_devices()               
        print('The current devices are available:\n')        
        for dev in self.available_devices:            
            print(dev)

    # set device
    #--------------------------------------------------------------------------
    def set_device(self):
       
        if CONFIG["training"]["ML_DEVICE"] == 'GPU':
            self.physical_devices = tf.config.list_physical_devices('GPU')
            if not self.physical_devices:
                print('\nNo GPU found. Falling back to CPU\n')
                tf.config.set_visible_devices([], 'GPU')
            else:
                if CONFIG["training"]["MIXED_PRECISION"]:
                    policy = keras.mixed_precision.Policy('mixed_float16')
                    keras.mixed_precision.set_global_policy(policy) 
                    print('\nMixed precision policy is active during training\n')
                tf.config.set_visible_devices(self.physical_devices[0], 'GPU')
                os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'                 
                print('\nGPU is set as active device\n')
                   
        elif CONFIG["training"]["ML_DEVICE"] == 'CPU':
            tf.config.set_visible_devices([], 'GPU')
            print('\nCPU is set as active device\n')    

    #--------------------------------------------------------------------------
    def training_session(self, model, train_data, validation_data, current_checkpoint_path):

        # initialize the real time history callback    
        RTH_callback = RealTimeHistory(current_checkpoint_path, validation=True)
        callbacks_list = [RTH_callback]

        # initialize tensorboard if requested    
        if CONFIG["training"]["USE_TENSORBOARD"]:
            log_path = os.path.join(current_checkpoint_path, 'tensorboard')
            callbacks_list.append(tf.keras.callbacks.TensorBoard(log_dir=log_path, 
                                                                 histogram_freq=1))

        # training loop and save model at end of training
        serializer = ModelSerializer() 
        num_processors = CONFIG["training"]["NUM_PROCESSORS"]  
        epochs = CONFIG["training"]["EPOCHS"] 
        multiprocessing = num_processors > 1
        training = model.fit(train_data, epochs=epochs, validation_data=validation_data, 
                            callbacks=callbacks_list, workers=num_processors, 
                            use_multiprocessing=multiprocessing)

        serializer.save_pretrained_model(model, current_checkpoint_path)

        # save model parameters in json files         
        parameters = {'picture_shape' : CONFIG["model"]["IMG_SHAPE"],                           
                      'augmentation' : CONFIG["dataset"]["IMG_AUGMENT"],              
                      'batch_size' : CONFIG["training"]["BATCH_SIZE"],
                      'learning_rate' : CONFIG["training"]["LEARNING_RATE"],
                      'epochs' : CONFIG["training"]["EPOCHS"],
                      'seed' : CONFIG["SEED"],
                      'tensorboard' : CONFIG["training"]["USE_TENSORBOARD"]}

        serializer.save_model_parameters(current_checkpoint_path, parameters)

        


      