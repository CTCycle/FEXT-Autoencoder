import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.utils.preprocessing import dataset_from_images
from FEXT.utils.models import Inference
from FEXT.config.pathfinder import INFERENCE_INPUT_PATH, INFERENCE_OUTPUT_PATH, CHECKPOINT_PATH
import FEXT.config.configurations as cnf


# [RUN MAIN]
if __name__ == '__main__':

    # 1. [EXTRACT FEATURES FROM IMAGES]
    #--------------------------------------------------------------------------
    inference = Inference(cnf.SEED) 

    # find and assign images path    
    df_images = dataset_from_images(INFERENCE_INPUT_PATH)

    # selected and load the pretrained model, then print the summary    
    inference = Inference(cnf.SEED) 
    model, parameters = inference.load_pretrained_model(CHECKPOINT_PATH)
    model.summary(expand_nested=True)

    # isolate the encoder from the autoencoder model, and use it for inference     
    encoder_input = model.get_layer('input_1')  
    encoder_output = model.get_layer('fe_xt_encoder')  
    encoder_model = keras.Model(inputs=encoder_input.input, outputs=encoder_output.output)

    # extract features from images using the encoder output    
    features = {}
    for pt in tqdm(df_images['path'].to_list()):
        try:
            image = inference.images_loader(pt, parameters['picture_shape'])
            image = tf.expand_dims(image, axis=0)
            extracted_features = encoder_model.predict(image, verbose=0)
            features.update({pt : extracted_features})
        except: 
            features.update({pt : 'Could not extract features'})

    # combine extracted features with images name and save them in numpy arrays    
    structured_data = np.array([(image, features[image]) for image in features], dtype=object)
    file_loc = os.path.join(INFERENCE_OUTPUT_PATH, 'extracted_features.npy')
    np.save(file_loc, structured_data)


