# [SETTING ENVIRONMENT VARIABLES]
from FEXT.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.validation.reports import DataAnalysisPDF
from FEXT.commons.utils.validation.images import ImageAnalysis
from FEXT.commons.utils.dataloader.serializer import DataSerializer
from FEXT.commons.utils.process.splitting import TrainValidationSplit
from FEXT.commons.constants import CONFIG, IMG_DATA_PATH
from FEXT.commons.logger import logger

# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD DATASET]
    #--------------------------------------------------------------------------  
    serializer = DataSerializer(CONFIG)     
    images_paths = serializer.get_images_path(IMG_DATA_PATH)  
    logger.info(f'The image dataset is composed of {len(images_paths)} images')  
    
    # 2. [COMPUTE IMAGE STATISTICS]
    #--------------------------------------------------------------------------
    analyzer = ImageAnalysis()
    logger.info('Calculating image statistics and generating dataset report')
    logger.info('Focusing on mean pixel values, pixel standard deviation, image noise ratio')
    image_statistics, images = analyzer.calculate_image_statistics(images_paths)    

    logger.info('Generating the pixel intensity histogram')
    analyzer.calculate_pixel_intensity(images)  

    # 3. [COMPARE TRAIN AND TEST DATASETS]
    #--------------------------------------------------------------------------
    splitter = TrainValidationSplit(images_paths, CONFIG)     
    train_data, validation_data = splitter.split_train_and_validation()
    logger.info('Splitting images pool into train and validation datasets')
    logger.info(f'Number of train samples: {len(train_data)}')
    logger.info(f'Number of validation samples: {len(validation_data)}')

    # 2. [INITIALIZE PDF REPORT]
    #--------------------------------------------------------------------------
    report = DataAnalysisPDF()
