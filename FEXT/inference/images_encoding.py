# [SET ML BACKEND]
import os 
os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.dataloader.serializer import get_images_path
from FEXT.commons.utils.dataloader.serializer import ModelSerializer
from FEXT.commons.utils.learning.inferencer import FeatureEncoding
from FEXT.commons.constants import CONFIG, ENCODED_INPUT_PATH
from FEXT.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [EXTRACT FEATURES FROM IMAGES]
    #--------------------------------------------------------------------------   
    # select a fraction of data for training
    images_paths = get_images_path(ENCODED_INPUT_PATH)

    # selected and load the pretrained model, then print the summary     
    logger.info('Loading specific checkpoint from pretrained models') 
    modelserializer = ModelSerializer()    
    model, configuration, history = modelserializer.load_pretrained_model()
    model.summary(expand_nested=True)    

    # extract features from images using the encoder output    
    encoder = FeatureEncoding(model, configuration)    
    encoder.encoder_images(images_paths)
