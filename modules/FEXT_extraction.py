import os
import sys
import pandas as pd
import tensorflow as tf
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
from modules.components.model_assets import Inference, DataGenerator
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

# select a fraction of data for training
#------------------------------------------------------------------------------
dataset = pd.DataFrame(images_paths, columns = ['images path'])

# define truncated model to get bottleneck layer outputs
#------------------------------------------------------------------------------
inference = Inference() 
model = inference.load_pretrained_model(GlobVar.model_path)
parameters = inference.model_configuration
model.summary(expand_nested=True)

encoder_layer = model.get_layer('encoder_output')
encoder_output = encoder_layer.output
encoder_model = Model(inputs=model.input, outputs=encoder_output)
encoder_model.summary()

# predict features
#------------------------------------------------------------------------------
features = []
for pt in images_paths:
    image = inference.images_loader(pt, parameters['Picture size'], 3)
    extracted_features = encoder_model.predict(image, verbose = 0)
    features.append(extracted_features)

# save data
#------------------------------------------------------------------------------
dataset['features'] = features
file_loc = os.path.join(GlobVar.pred_path, 'images_dataset.csv')  
dataset.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')


