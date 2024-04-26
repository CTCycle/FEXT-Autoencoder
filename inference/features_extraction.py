import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

# setting warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add parent folder path to the namespace
#------------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

# import modules and components
#------------------------------------------------------------------------------ 
from utils.preprocessing import dataset_from_images
from utils.models import Inference
import utils.pathfinder as globpt
import configurations as cnf

# specify relative paths from global paths and create subfolders
#------------------------------------------------------------------------------
images_path = os.path.join(globpt.inference_path, 'images')
cp_path = os.path.join(globpt.train_path, 'checkpoints')
os.mkdir(images_path) if not os.path.exists(images_path) else None
os.mkdir(cp_path) if not os.path.exists(cp_path) else None

# [INFERENCE]
#==============================================================================
# module for the selection of different operations
#==============================================================================
print('''
-------------------------------------------------------------------------------
Features Extraction: extraction from pretrained model
-------------------------------------------------------------------------------
.... 
''')

inference = Inference(cnf.seed) 

# find and assign images path
#------------------------------------------------------------------------------
df_images = dataset_from_images(images_path)

# selected and load the pretrained model, then print the summary
#------------------------------------------------------------------------------
inference = Inference(cnf.seed) 
model, parameters = inference.load_pretrained_model(cp_path)
model.summary(expand_nested=True)

# isolate the encoder from the autoencoder model, and use it for inference 
#------------------------------------------------------------------------------
encoder_input = model.get_layer('input_1')  
encoder_output = model.get_layer('fe_xt_encoder')  
encoder_model = keras.Model(inputs=encoder_input.input, outputs=encoder_output.output)

# extract features from images using the encoder output
#------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------
structured_data = np.array([(image, features[image]) for image in features], dtype=object)
file_loc = os.path.join(globpt.inference_path, 'extracted_features.npy')
np.save(file_loc, structured_data)


