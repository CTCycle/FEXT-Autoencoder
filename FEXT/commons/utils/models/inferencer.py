import os
import numpy as np
import torch
import tensorflow as tf
import keras

from FEXT.commons.utils.dataloader.serializer import DataSerializer
from FEXT.commons.constants import CONFIG, ENCODED_OUTPUT_PATH
from FEXT.commons.logger import logger



# [INFERENCE]
###############################################################################
class FeatureExtractor:
    
    def __init__(self, model : keras.Model):
       
        np.random.seed(CONFIG["SEED"])
        torch.manual_seed(CONFIG["SEED"])
        self.dataserializer = DataSerializer()        
        # isolate the encoder from the autoencoder model, and use it for inference     
        encoder_input = model.get_layer('input_layer').output
        encoder_output = model.get_layer('fe_xt_encoder').output 
        self.encoder_model = keras.Model(inputs=encoder_input, outputs=encoder_output)        

    #--------------------------------------------------------------------------
    def extract_from_encoder(self, images_paths, parameters):
        
        features = {}
        for pt in images_paths:
            try:
                image = self.dataserializer.load_image(pt, parameters['picture_shape'])
                image = tf.expand_dims(image, axis=0)
                extracted_features = self.encoder_model.predict(image, verbose=0)
                features.update({pt : extracted_features})
            except: 
                features.update({pt : 'Could not extract features'})
                logger.warning(f'Could not extract features from image at {pt}')

        # combine extracted features with images name and save them in numpy arrays    
        structured_data = np.array([(image, features[image]) for image in features], dtype=object)
        file_loc = os.path.join(ENCODED_OUTPUT_PATH, 'extracted_features.npy')
        np.save(file_loc, structured_data)

        logger.debug(f'Extracted img features saved as numpy array at {file_loc}')

        return features





