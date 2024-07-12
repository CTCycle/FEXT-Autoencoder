# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.dataloader.serializer import get_images_path, DataSerializer
from FEXT.commons.utils.dataloader.serializer import DataSerializer, ModelSerializer
from FEXT.commons.utils.models.inferencer import FeatureExtractor
from FEXT.commons.constants import ENCODED_INPUT_PATH
from FEXT.commons.logger import logger

# [RUN MAIN]
if __name__ == '__main__':

    # 1. [EXTRACT FEATURES FROM IMAGES]
    #--------------------------------------------------------------------------    
    dataserializer = DataSerializer()   
    modelserializer = ModelSerializer()     
    
    # select a fraction of data for training
    images_paths = get_images_path(ENCODED_INPUT_PATH)

    # selected and load the pretrained model, then print the summary     
    logger.info('Loading specific checkpoint from pretrained models')   
    model, parameters = modelserializer.load_pretrained_model()
    model.summary(expand_nested=True)    

    # extract features from images using the encoder output    
    extractor = FeatureExtractor(model)
    extractor.extract_from_encoder(images_paths, parameters)
