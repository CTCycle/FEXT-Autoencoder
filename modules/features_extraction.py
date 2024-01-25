import os
import sys
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from tqdm import tqdm

# setting warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add modules path if this file is launched as __main__
#------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  

# import modules and components
#------------------------------------------------------------------------------ 
from modules.components.model_assets import Inference
import modules.global_variables as GlobVar
import configurations as cnf


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

# find and assign images path
#------------------------------------------------------------------------------
images_paths = []
for root, dirs, files in os.walk(GlobVar.pred_path):
    for file in files:
        images_paths.append(os.path.join(root, file))

# load the model for inference and print summary
#------------------------------------------------------------------------------
inference = Inference() 
model = inference.load_pretrained_model(GlobVar.model_path)
parameters = inference.model_configuration
model.summary(expand_nested=True)

# define truncated model to get the encoder output only
#------------------------------------------------------------------------------
encoder_input = model.get_layer('input_1')  
encoder_output = model.get_layer('fe_xt_encoder')  
encoder_model = keras.Model(inputs=encoder_input.input, outputs=encoder_output.output)

# predict features
#------------------------------------------------------------------------------
features = {}
for pt in tqdm(images_paths):
    try:
        image = inference.images_loader(pt, parameters['Picture size'], 3)
        image = tf.expand_dims(image, axis=0)
        extracted_features = encoder_model.predict(image, verbose=0)
        features.update({pt : extracted_features})
    except: 
        features.update({pt : 'Could not extract features'})

# save data
#------------------------------------------------------------------------------
dataset = pd.DataFrame(list(features.items()), columns=['Images', 'Features'])
file_loc = os.path.join(GlobVar.pred_path, 'images_dataset.csv')  
dataset.to_csv(file_loc, index=False, sep=';', encoding='utf-8')


