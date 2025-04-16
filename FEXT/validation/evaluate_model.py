import random 

# [SETTING ENVIRONMENT VARIABLES]
from FEXT.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.data.loader import InferenceDataLoader
from FEXT.commons.utils.data.serializer import DataSerializer, ModelSerializer
from FEXT.commons.utils.data.splitting import TrainValidationSplit
from FEXT.commons.utils.inference.encoding import ImageEncoding
from FEXT.commons.utils.validation.images import ImageReconstruction
from FEXT.commons.utils.validation.checkpoints import ModelEvaluationSummary
from FEXT.commons.utils.validation.reports import evaluation_report
from FEXT.commons.constants import CONFIG, IMG_PATH
from FEXT.commons.logger import logger

# [RUN MAIN]
###############################################################################
if __name__ == '__main__':        

    # 1. [LOAD DATASET]
    #--------------------------------------------------------------------------  
    summarizer = ModelEvaluationSummary(CONFIG)    
    checkpoints_summary = summarizer.checkpoints_summary() 
    logger.info(f'Checkpoints summary has been created for {checkpoints_summary.shape[0]} models')  
    
    # 2. [LOAD MODEL]
    #--------------------------------------------------------------------------
    # selected and load the pretrained model, then print the summary 
    modelserializer = ModelSerializer()         
    model, configuration, checkpoint_path = modelserializer.select_and_load_checkpoint()
    model.summary(expand_nested=True)

    # isolate the encoder from the autoencoder model   
    encoder = ImageEncoding(model, configuration, checkpoint_path)
    encoder_model = encoder.encoder_model   

    # 3. [LOAD AND SPLIT DATA]
    #--------------------------------------------------------------------------
    dataserializer = DataSerializer(configuration)
    images_path = dataserializer.get_images_path_from_directory(IMG_PATH)

    splitter = TrainValidationSplit(images_path, configuration)     
    _, validation_images = splitter.split_train_and_validation() 

    # 4. [EVALUATE ON TRAIN AND VALIDATION]
    #--------------------------------------------------------------------------  
    # use tf.data.Dataset to build the model dataloader with a larger batch size
    # the dataset is built on top of the training and validation data
    loader = InferenceDataLoader(configuration)    
    validation_dataset = loader.build_inference_dataloader(validation_images)

    # evaluate model performance over the training and validation dataset    
    evaluation_report(model, validation_dataset) 

    # 5. [COMPARE RECONTRUCTED IMAGES]
    #--------------------------------------------------------------------------   
    validator = ImageReconstruction(CONFIG, model, checkpoint_path)      
    validator.visualize_reconstructed_images(validation_images)    
   

   
