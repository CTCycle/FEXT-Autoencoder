import os
import sys
import pandas as pd

# setting warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add parent folder path to the namespace
#------------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import modules and components
#------------------------------------------------------------------------------
from components.data_assets import PreProcessing 
from components.validation_assets import DataValidation
import components.global_paths as globpt
import configurations as cnf

# specify relative paths from global paths and create subfolders
#------------------------------------------------------------------------------
images_path = os.path.join(globpt.data_path, 'images') 
val_path = os.path.join(globpt.data_path, 'validation')
os.mkdir(images_path) if not os.path.exists(images_path) else None
os.mkdir(val_path) if not os.path.exists(val_path) else None  


# [LOAD AND PREPARE DATA]
#==============================================================================
#==============================================================================
print('''
-------------------------------------------------------------------------------
Data Validation
-------------------------------------------------------------------------------
''')

# initialize the preprocessing and validation class
#------------------------------------------------------------------------------
preprocessor = PreProcessing()
validator = DataValidation()

# find and assign images path
#------------------------------------------------------------------------------
images_paths = []
for root, dirs, files in os.walk(images_path):
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
#==============================================================================
print('''Generating pixel intensity histograms (train vs test datasets)
''')

# load train and test images as numpy arrays
#------------------------------------------------------------------------------
train_images = preprocessor.load_images(train_data['images path'], cnf.picture_shape[:-1], 
                                        as_tensor=False,  normalize=False)
test_images = preprocessor.load_images(test_data['images path'], cnf.picture_shape[:-1], 
                                       as_tensor=False, normalize=False)

# validate pixel intensity histograms for both datasets
#------------------------------------------------------------------------------
validator.pixel_intensity_histograms(train_images, test_images, val_path,
                                     names=['Train', 'Test'])

# Print report with info about the data evaluation 
#------------------------------------------------------------------------------
print(f'''
-------------------------------------------------------------------------------
Data validation report
-------------------------------------------------------------------------------
Number of train samples: {train_data.shape[0]}
Number of test samples:  {test_data.shape[0]}
-------------------------------------------------------------------------------
''')
