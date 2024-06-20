import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.preprocessing import get_images_path
from FEXT.commons.utils.dataloader.serializer import DataSerializer, ModelSerializer
from FEXT.commons.utils.models.inferencer import FeatureExtractor
from FEXT.commons.pathfinder import CHECKPOINT_PATH
import FEXT.commons.configurations as cnf


# [RUN MAIN]
if __name__ == '__main__':

    # 1. [EXTRACT FEATURES FROM IMAGES]
    #--------------------------------------------------------------------------
    extractor = FeatureExtractor(cnf.SEED)
    dataserializer = DataSerializer()   
    modelserializer = ModelSerializer() 
    
    
    # select a fraction of data for training
    images_paths = get_images_path()

    # selected and load the pretrained model, then print the summary        
    model, parameters = modelserializer.load_pretrained_model(CHECKPOINT_PATH)
    model.summary(expand_nested=True)

    # isolate the encoder from the autoencoder model, and use it for inference     
    encoder_input = model.get_layer('input_1')  
    encoder_output = model.get_layer('fe_xt_encoder')  
    encoder_model = keras.Model(inputs=encoder_input.input, outputs=encoder_output.output)

    # extract features from images using the encoder output    
    extractor.extract_from_encoder(images_paths, encoder_model)
