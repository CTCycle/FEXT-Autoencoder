import os

# [SETTING ENVIRONMENT VARIABLES]
from FEXT.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.data.serializer import DataSerializer, ModelSerializer
from FEXT.commons.utils.inference.encoding import ImageEncoding
from FEXT.commons.constants import CONFIG, INFERENCE_INPUT_PATH
from FEXT.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [GET IMAGES]
    #--------------------------------------------------------------------------   
    # select a fraction of data for training
    dataserializer = DataSerializer(CONFIG)
    images_paths = dataserializer.get_images_path(INFERENCE_INPUT_PATH)

    # 2. [LOAD PRETRAINED MODEL]
    #--------------------------------------------------------------------------
    # selected and load the pretrained model, then print the summary   
    modelserializer = ModelSerializer()    
    model, configuration, history, checkpoint_path = modelserializer.select_and_load_checkpoint()
    model.summary(expand_nested=True)   

    # 3. [ENCODE IMAGES]
    #--------------------------------------------------------------------------    
    logger.info(f'Start encoding images using model {os.path.basename(checkpoint_path)}')    
    logger.info(f'{len(images_paths)} images have been found') 
    # extract features from images using the encoder output    
    encoder = ImageEncoding(model, configuration, checkpoint_path)    
    encoder.encode_images_features(images_paths)
    logger.info('Encoded images have been saved as .npy')

