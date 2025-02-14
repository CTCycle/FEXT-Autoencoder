# [SET ML BACKEND]
import os 
os.environ["KERAS_BACKEND"] = "torch"

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.dataloader.serializer import DataSerializer, ModelSerializer
from FEXT.commons.utils.inference.encoding import ImageEncoding
from FEXT.commons.constants import CONFIG, ENCODED_INPUT_PATH
from FEXT.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [EXTRACT FEATURES FROM IMAGES]
    #--------------------------------------------------------------------------   
    # select a fraction of data for training
    dataserializer = DataSerializer(CONFIG)
    images_paths = dataserializer.get_images_path(ENCODED_INPUT_PATH)

    # selected and load the pretrained model, then print the summary   l
    modelserializer = ModelSerializer()    
    model, configuration, history, checkpoint_path = modelserializer.select_and_load_checkpoint()
    model.summary(expand_nested=True)   

    # extract features from images using the encoder output 
    logger.info(f'Start encoding images using model {os.path.basename(checkpoint_path)}')
    logger.info(f'{len(images_paths)} images have been found in resources/encoding/images')    
    encoder = ImageEncoding(model, configuration)    
    encoder.encode_images_features(images_paths)
    logger.info(f'Extracted images features have been saved as .npy in resources/encoding')

