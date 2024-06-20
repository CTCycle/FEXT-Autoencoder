from tensorflow import keras

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.dataloader.serializer import get_images_path, DataSerializer
from FEXT.commons.utils.dataloader.serializer import DataSerializer, ModelSerializer
from FEXT.commons.utils.models.inferencer import FeatureExtractor
from FEXT.commons.pathfinder import ENCODED_INPUT_PATH

# [RUN MAIN]
if __name__ == '__main__':

    # 1. [EXTRACT FEATURES FROM IMAGES]
    #--------------------------------------------------------------------------    
    dataserializer = DataSerializer()   
    modelserializer = ModelSerializer()     
    
    # select a fraction of data for training
    images_paths = get_images_path(ENCODED_INPUT_PATH)

    # selected and load the pretrained model, then print the summary     
    print('\nLoading specific checkpoint from pretrained models\n')   
    model, parameters = modelserializer.load_pretrained_model()
    model.summary(expand_nested=True)

    # isolate the encoder from the autoencoder model, and use it for inference     
    encoder_input = model.get_layer('input_1')  
    encoder_output = model.get_layer('fe_xt_encoder')  
    encoder_model = keras.Model(inputs=encoder_input.input, outputs=encoder_output.output)

    # extract features from images using the encoder output    
    extractor = FeatureExtractor(model)
    extractor.extract_from_encoder(images_paths, encoder_model)
