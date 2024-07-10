# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.dataloader.generators import build_tensor_dataset
from FEXT.commons.utils.dataloader.serializer import get_images_path, DataSerializer, ModelSerializer
from FEXT.commons.utils.preprocessing import DataSplit
from FEXT.commons.utils.models.training import ModelTraining
from FEXT.commons.utils.models.autoencoder import FeXTAutoEncoder
from FEXT.commons.constants import CONFIG, IMG_DATA_PATH
from FEXT.commons.logger import logger


# [RUN MAIN]
if __name__ == '__main__':

    # 1. [LOAD AND SPLIT DATA]
    #--------------------------------------------------------------------------    
    # select a fraction of data for training 
    sample_size = CONFIG["dataset"]["SAMPLE_SIZE"]   
    images_paths = get_images_path(IMG_DATA_PATH, sample_size=sample_size)    

    # split data
    logger.info('Preparing dataset of images based on splitting sizes')  
    splitter = DataSplit(images_paths)     
    train_data, validation_data = splitter.split_data()   

    # create subfolder for preprocessing data    
    logger.info('Saving images path references') 
    dataserializer = DataSerializer()
    model_folder_path = dataserializer.create_checkpoint_folder()
    dataserializer.save_preprocessed_data(train_data, validation_data, 
                                          model_folder_path)      

    # 2. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
    #--------------------------------------------------------------------------
    # initialize training device 
    # allows changing device prior to initializing the generators
    logger.info('Building autoencoder model and data loaders')     
    trainer = ModelTraining()
    trainer.set_device()

    # initialize the TensorDataSet class with the generator instances
    # create the tf.datasets using the previously initialized generators    
    train_dataset = build_tensor_dataset(train_data)
    validation_dataset = build_tensor_dataset(validation_data)
    
    # 3. [TRAINING MODEL]  
    #--------------------------------------------------------------------------  
    # Setting callbacks and training routine for the features extraction model 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/ 
            
    logger.info('FeXT training report')
    logger.info('--------------------------------------------------------------')    
    logger.info(f'Number of train samples:       {len(train_data)}')
    logger.info(f'Number of validation samples:  {len(validation_data)}')      
    logger.info(f'Picture shape:                 {CONFIG["model"]["IMG_SHAPE"]}')   
    logger.info(f'Batch size:                    {CONFIG["training"]["BATCH_SIZE"]}')
    logger.info(f'Epochs:                        {CONFIG["training"]["EPOCHS"]}')  
    logger.info('--------------------------------------------------------------')  

    # build the autoencoder model     
    autoencoder = FeXTAutoEncoder()
    model = autoencoder.get_model(summary=True) 

    # generate graphviz plot fo the model layout 
    modelserializer = ModelSerializer()     
    modelserializer.save_model_plot(model, model_folder_path)              

    # perform training and save model at the end
    trainer.train_model(model, train_dataset, validation_dataset, model_folder_path)



