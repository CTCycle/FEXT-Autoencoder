# [SET KERAS BACKEND]
import os 
os.environ["KERAS_BACKEND"] = "torch"

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.dataloader.generators import build_model_dataloader
from FEXT.commons.utils.dataloader.serializer import DataSerializer, ModelSerializer
from FEXT.commons.utils.process.splitting import TrainValidationSplit
from FEXT.commons.utils.learning.training import ModelTraining
from FEXT.commons.utils.validation.reports import log_training_report
from FEXT.commons.constants import CONFIG, IMG_DATA_PATH
from FEXT.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD PRETRAINED MODEL]
    #--------------------------------------------------------------------------     
    # selected and load the pretrained model, then print the summary     
    logger.info('Loading specific checkpoint from pretrained models') 
    modelserializer = ModelSerializer()   
    model, configuration, history, checkpoint_path = modelserializer.select_and_load_checkpoint()    
    model.summary(expand_nested=True)  
    
    # setting device for training    
    trainer = ModelTraining(configuration)    
    trainer.set_device()  

    # 2. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
    # initialize training device, allows changing device prior to initializing the generators
    #--------------------------------------------------------------------------   
    # select a fraction of data for training
    dataserializer = DataSerializer(configuration) 
    images_path = dataserializer.load_data_from_checkpoint(checkpoint_path)    
    
    # split data into train and validation        
    logger.info('Preparing dataset of images based on splitting sizes')  
    splitter = TrainValidationSplit(images_path, configuration)     
    train_data, validation_data = splitter.split_train_and_validation() 

    # initialize the TensorDataSet class with the generator instances
    # create the tf.datasets using the previously initialized generators   
    logger.info('Building data loaders') 
    train_dataset, validation_dataset = build_model_dataloader(train_data, validation_data, configuration)
    
    # 3. [TRAINING MODEL]
    # Setting callbacks and training routine for the features extraction model 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/ 
    #--------------------------------------------------------------------------
    log_training_report(train_data, validation_data, configuration, from_checkpoint=True) 
                        
    # resume training from pretrained model    
    trainer.train_model(model, train_dataset, validation_dataset, checkpoint_path,
                        from_checkpoint=True)



