# [SETTING ENVIRONMENT VARIABLES]
from FEXT.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.data.tensordata import TrainingDatasetBuilder
from FEXT.commons.utils.data.serializer import DataSerializer, ModelSerializer
from FEXT.commons.utils.process.splitting import TrainValidationSplit
from FEXT.commons.utils.learning.training import ModelTraining
from FEXT.commons.utils.learning.autoencoder import FeXTAutoEncoder
from FEXT.commons.utils.validation.reports import log_training_report
from FEXT.commons.constants import CONFIG, IMG_PATH
from FEXT.commons.logger import logger

# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD AND SPLIT DATA]
    #--------------------------------------------------------------------------    
    # select a fraction of data for training
    dataserializer = DataSerializer(CONFIG)     
    images_paths = dataserializer.get_images_path(IMG_PATH)    

    # split data into train and validation        
    logger.info('Preparing dataset of images based on splitting sizes')  
    splitter = TrainValidationSplit(images_paths, CONFIG)     
    train_data, validation_data = splitter.split_train_and_validation()     
    
    # 2. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
    #-------------------------------------------------------------------------- 
    modelserializer = ModelSerializer()
    checkpoint_path = modelserializer.create_checkpoint_folder()

    logger.info('Setting device for training operations based on user configurations') 
    trainer = ModelTraining(CONFIG)
    trainer.set_device()   

    # create the tf.datasets using the previously initialized generators 
    logger.info('Building model data loaders with prefetching and parallel processing') 
    builder = TrainingDatasetBuilder(CONFIG)   
    train_dataset, validation_dataset = builder.build_model_dataloader(
        train_data, validation_data)           
    
    # 3. [TRAINING MODEL]
    # Setting callbacks and training routine for the features extraction model 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/ 
    #--------------------------------------------------------------------------
    log_training_report(train_data, validation_data, CONFIG)

    # build the autoencoder model 
    logger.info('Building FeXT AutoEncoder model based on user configurations')      
    autoencoder = FeXTAutoEncoder(CONFIG)
    model = autoencoder.get_model(model_summary=True)
    
    # generate graphviz plot for the model layout        
    modelserializer.save_model_plot(model, checkpoint_path)              

    # perform training and save model at the end
    logger.info('Starting FeXT AutoEncoder training') 
    trainer.train_model(model, train_dataset, validation_dataset, checkpoint_path)



