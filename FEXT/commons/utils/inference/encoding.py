import os
import numpy as np
import keras
from tqdm import tqdm

from FEXT.commons.utils.data.loader import InferenceDataLoader
from FEXT.commons.constants import INFERENCE_PATH
from FEXT.commons.logger import logger


# [INFERENCE]
###############################################################################
class ImageEncoding:
    
    def __init__(self, model : keras.Model, configuration : dict, checkpoint_path : str):       
        keras.utils.set_random_seed(configuration.get('train_seed', 42)) 
        self.dataloader = InferenceDataLoader(configuration)
        self.checkpoint_name = os.path.basename(checkpoint_path)        
        self.configuration = configuration
        self.model = model 

        # isolate the encoder submodel from the autoencoder model             
        encoder_output = model.get_layer('compression_layer').output 
        self.encoder_model = keras.Model(
            inputs=model.input, outputs=encoder_output)              

    #--------------------------------------------------------------------------
    def encode_images_features(self, images_paths, progress_callback=None):        
        features = {}
        for i, pt in enumerate(tqdm(images_paths, desc='Encoding images', total=len(images_paths))):
            image_name = os.path.basename(pt)
            try:
                image = self.dataloader.load_image_as_array(pt)
                image = np.expand_dims(image, axis=0)
                extracted_features = self.encoder_model.predict(image, verbose=0)
                features[pt] = extracted_features
            except Exception as e:
                features[pt] = f'Error during encoding: {str(e)}'
                logger.error(f'Could not encode image {image_name}: {str(e)}')
            
            if progress_callback is not None:
                total = len(images_paths)
                percent = int((i + 1) * 100 / total)
                progress_callback(percent)

        # combine extracted features with images name and save them in numpy arrays    
        structured_data = np.array(
            [(image, features[image]) for image in features], dtype=object)
        file_loc = os.path.join(
            INFERENCE_PATH, f'encoded_images_{self.checkpoint_name}.npy')
        np.save(file_loc, structured_data)
        
        return features





