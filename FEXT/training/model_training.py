# [SET KERAS BACKEND]
import os 
os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.dataloader.generators import ML_model_dataloader
from FEXT.commons.utils.dataloader.serializer import get_images_path, DataSerializer, ModelSerializer
from FEXT.commons.utils.process.splitting import TrainValidationSplit
from FEXT.commons.utils.learning.training import ModelTraining
from FEXT.commons.utils.learning.autoencoder import FeXTAutoEncoder
from FEXT.commons.utils.validation.reports import log_training_report
from FEXT.commons.constants import CONFIG, IMG_DATA_PATH
from FEXT.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD AND SPLIT DATA]
    #--------------------------------------------------------------------------    
    # select a fraction of data for training     
    images_paths = get_images_path(IMG_DATA_PATH, CONFIG)    

    # split data into train and validation        
    logger.info('Preparing dataset of images based on splitting sizes')  
    splitter = TrainValidationSplit(images_paths, CONFIG)     
    train_data, validation_data = splitter.split_train_and_validation()   

    # create subfolder for preprocessing data    
    modelserializer = ModelSerializer()
    checkpoint_path = modelserializer.create_checkpoint_folder() 

    # save preprocessed data references
    logger.info(f'Saving images references in {checkpoint_path}')
    dataserializer = DataSerializer(CONFIG)
    dataserializer.save_preprocessed_data(train_data, validation_data, checkpoint_path)    

    # 2. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
    #--------------------------------------------------------------------------
    # initialize training device, allows changing device prior to initializing the generators
    #--------------------------------------------------------------------------
    logger.info('Building autoencoder model and data loaders')     
    trainer = ModelTraining(CONFIG)
    trainer.set_device()   

    # initialize the TensorDataSet class with the generator instances
    # create the tf.datasets using the previously initialized generators    
    train_dataset, validation_dataset = ML_model_dataloader(train_data, validation_data, CONFIG)         
    
    # 3. [TRAINING MODEL]  
    #--------------------------------------------------------------------------  
    # Setting callbacks and training routine for the features extraction model 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/ 
    #--------------------------------------------------------------------------
    log_training_report(train_data, validation_data, CONFIG)

    # build the autoencoder model     
    autoencoder = FeXTAutoEncoder(CONFIG)
    model = autoencoder.get_model(model_summary=True)
    
    # generate graphviz plot for the model layout        
    modelserializer.save_model_plot(model, checkpoint_path)              

    # perform training and save model at the end
    trainer.train_model(model, train_dataset, validation_dataset, checkpoint_path)



