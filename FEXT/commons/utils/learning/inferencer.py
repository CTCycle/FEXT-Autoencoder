import os
import numpy as np
import keras
from tqdm import tqdm

from FEXT.commons.utils.dataloader.serializer import get_images_path
from FEXT.commons.utils.dataloader.serializer import DataSerializer
from FEXT.commons.constants import ENCODED_INPUT_PATH, ENCODED_PATH
from FEXT.commons.logger import logger


# [INFERENCE]
###############################################################################
class ImagesEncoding:
    
    def __init__(self, configuration):
       
        keras.utils.set_random_seed(configuration["SEED"])  
        self.dataserializer = DataSerializer(configuration)  
        self.img_shape = configuration["model"]["IMG_SHAPE"]
        self.configuration = None       
        self.encoder_model = None

    #--------------------------------------------------------------------------
    def get_encoder_from_checkpoint(self, model : keras.Model):  
        encoder_output = model.get_layer('compression_layer').output 
        self.encoder_model = keras.Model(inputs=model.input, outputs=encoder_output)            

    #--------------------------------------------------------------------------
    def single_image_encoding(self, path):
        
        image = self.dataserializer.load_image(path, self.img_shape)
        image = keras.ops.expand_dims(image, axis=0)
        extracted_features = self.encoder_model.predict(image, verbose=0)

        return extracted_features
    
    #--------------------------------------------------------------------------
    def encode_images(self):

        images_path = get_images_path(ENCODED_INPUT_PATH)
        encodings = {}
        for path in images_path:
            encoded_img = self.single_image_encoding(path)
            encodings[os.path.basename(path)] = encoded_img

        # combine extracted features with images name and save them in numpy arrays    
        structured_data = np.array([(image, encodings[image]) for image in images_path], dtype=object)
        file_loc = os.path.join(ENCODED_PATH, 'extracted_features.npy')
        np.save(file_loc, structured_data)

        logger.debug(f'Extracted img features saved as numpy array at {file_loc}')

        return structured_data
        

        





