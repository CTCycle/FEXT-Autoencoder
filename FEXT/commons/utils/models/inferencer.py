import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from FEXT.commons.utils.dataloader.serializer import DataSerializer
from FEXT.commons.constants import CONFIG, ENCODED_OUTPUT_PATH



# [INFERENCE]
#------------------------------------------------------------------------------
class FeatureExtractor:
    
    def __init__(self, model):
        
        np.random.seed(CONFIG["SEED"])
        tf.random.set_seed(CONFIG["SEED"])
        self.dataserializer = DataSerializer()
        self.model = model

    def extract_from_encoder(self, images_paths, parameters):
        
        features = {}
        for pt in images_paths:
            try:
                image = self.dataserializer.load_images(pt, parameters['picture_shape'])
                image = tf.expand_dims(image, axis=0)
                extracted_features = self.model.predict(image, verbose=0)
                features.update({pt : extracted_features})
            except: 
                features.update({pt : 'Could not extract features'})

        # combine extracted features with images name and save them in numpy arrays    
        structured_data = np.array([(image, features[image]) for image in features], dtype=object)
        file_loc = os.path.join(ENCODED_OUTPUT_PATH, 'extracted_features.npy')
        np.save(file_loc, structured_data)

        return features





