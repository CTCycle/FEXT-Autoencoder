# [SET KERAS BACKEND]
import os 
os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.dataloader.generators import training_data_pipeline
from FEXT.commons.utils.dataloader.serializer import DataSerializer, ModelSerializer
from FEXT.commons.utils.learning.training import ModelTraining
from FEXT.commons.utils.validation.reports import log_training_report
from FEXT.commons.constants import CONFIG
from FEXT.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD PRETRAINED MODEL]
    #--------------------------------------------------------------------------    
    dataserializer = DataSerializer()   
    modelserializer = ModelSerializer()     
    
    # selected and load the pretrained model, then print the summary     
    logger.info('Loading specific checkpoint from pretrained models')   
    model, configuration, history = modelserializer.load_pretrained_model()
    model_folder = modelserializer.loaded_model_folder
    model.summary(expand_nested=True)  
    
    # setting device for training    
    trainer = ModelTraining(configuration)    
    trainer.set_device()  

    # 2. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
    #--------------------------------------------------------------------------
    # initialize training device, allows changing device prior to initializing the generators
    #--------------------------------------------------------------------------   
    # load saved tf.datasets from the proper folders in the checkpoint directory     
    train_data, validation_data = dataserializer.load_preprocessed_data(model_folder)

    # initialize the TensorDataSet class with the generator instances
    # create the tf.datasets using the previously initialized generators   
    logger.info('Building data loaders') 
    train_dataset, validation_dataset = training_data_pipeline(train_data, validation_data, configuration)
    
    # 3. [TRAINING MODEL]  
    #--------------------------------------------------------------------------  
    # Setting callbacks and training routine for the features extraction model 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/ 
    #--------------------------------------------------------------------------
    log_training_report(train_data, validation_data, configuration, 
                        additional_epochs=CONFIG['TRAINING']['ADDITIONAL_EPOCHS'])

    # resume training from pretrained model    
    trainer.train_model(model, train_dataset, validation_dataset, model_folder,
                        from_checkpoint=True)



