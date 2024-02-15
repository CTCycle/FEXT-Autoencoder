import os
import sys
import pandas as pd
import tensorflow as tf
from keras.utils import plot_model

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
from modules.components.data_assets import PreProcessing, DataValidation
import modules.global_variables as GlobVar
import configurations as cnf

# [LOAD DATA]
#==============================================================================
# Load the csv with data and transform the tokenized text column to convert the
# strings into a series of integers
#==============================================================================
print('''
-------------------------------------------------------------------------------
Data Validation
-------------------------------------------------------------------------------
''')

preprocessor = PreProcessing()

# find and assign images path
#------------------------------------------------------------------------------
images_paths = []
for root, dirs, files in os.walk(GlobVar.images_path):
    for file in files:
        images_paths.append(os.path.join(root, file))

# select a fraction of data for training
#------------------------------------------------------------------------------
total_samples = cnf.num_train_samples + cnf.num_test_samples
df_images = pd.DataFrame(images_paths, columns=['images path'])
df_images = df_images.sample(total_samples, random_state=36)

# create train and test datasets (for validation)
#------------------------------------------------------------------------------
test_data = df_images.sample(n=cnf.num_test_samples, random_state=36)
train_data = df_images.drop(test_data.index)

# [DATA EVALUATION]
#==============================================================================
# ...
#==============================================================================
print('''Generating pixel intensity histograms (train vs test datasets)
''')

validator = DataValidation()

# load train and test images as numpy arrays
#------------------------------------------------------------------------------
train_images = preprocessor.load_images(train_data['images path'], cnf.picture_shape[:-1], 
                                        as_tensor=False,  normalize=False)
test_images = preprocessor.load_images(test_data['images path'], cnf.picture_shape[:-1], 
                                       as_tensor=False, normalize=False)

# validate pixel intensity histograms for both datasets
#------------------------------------------------------------------------------
validator.pixel_intensity_histograms(train_images, test_images, GlobVar.val_path,
                                     names=['Train', 'Test'])




# [TRAINING MODEL]
#==============================================================================
# Setting callbacks and training routine for the features extraction model. 
# use command prompt on the model folder and (upon activating environment), 
# use the bash command: python -m tensorboard.main --logdir tensorboard/
#==============================================================================

# Print report with info about the training parameters
#------------------------------------------------------------------------------
print(f'''
-------------------------------------------------------------------------------
Data validation report
-------------------------------------------------------------------------------
Number of train samples: {train_data.shape[0]}
Number of test samples:  {test_data.shape[0]}
-------------------------------------------------------------------------------
''')
