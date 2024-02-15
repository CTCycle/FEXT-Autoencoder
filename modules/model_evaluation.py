import os
import sys
import pandas as pd
import tensorflow as tf

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
from modules.components.model_assets import ModelValidation, Inference
import modules.global_variables as GlobVar
import configurations as cnf

# [LOAD MODEL AND DATA]
#==============================================================================
# Load data and models
#==============================================================================

# load the model for inference and print summary
#------------------------------------------------------------------------------
inference = Inference() 
model, parameters = inference.load_pretrained_model(GlobVar.models_path)
model_path = inference.folder_path
model.summary(expand_nested=True)

# load data
#------------------------------------------------------------------------------
filepath = os.path.join(model_path, 'preprocessing', 'train_X.csv')                
train_data = pd.read_csv(filepath, sep= ';', encoding='utf-8')
filepath = os.path.join(model_path, 'preprocessing', 'test_X.csv')                
test_data = pd.read_csv(filepath, sep= ';', encoding='utf-8')

# [MODEL VALIDATION]
#==============================================================================
# ...
#==============================================================================
print('''
-------------------------------------------------------------------------------
MODEL EVALUATION
-------------------------------------------------------------------------------
A set of images is compared to show difference between the input images and the 
reconstructed images, to be considered as a visual inspection of the model performance
''')

validator = ModelValidation(model)

# create subfolder for evaluation data
#------------------------------------------------------------------------------
eval_path = os.path.join(model_path, 'evaluation') 
if not os.path.exists(eval_path):
    os.mkdir(eval_path)

# predict images from train and test subsets
#------------------------------------------------------------------------------
train_eval = model.evaluate(train_data)
test_eval = model.evaluate(test_data)

print(f'''
-------------------------------------------------------------------------------
MODEL EVALUATION
-------------------------------------------------------------------------------    
Train dataset:
- Loss:   {train_eval[0]}
- Metric: {train_eval[1]} 

Test dataset:
- Loss:   {test_eval[0]}
- Metric: {test_eval[1]}        
''')

# perform visual validation for the train dataset (initialize a validation tf.dataset
# with batch size of 10 images)
#------------------------------------------------------------------------------
validation_batch = train_data.unbatch().batch(10).take(1)
for images, labels in validation_batch:
    recostructed_images = model.predict(images, verbose=0)
    validator.visual_validation(images, recostructed_images, 'visual_validation_train', 
                                eval_path)

# perform visual validation for the test dataset (initialize a validation tf.dataset
# with batch size of 10 images)
#------------------------------------------------------------------------------
validation_batch = test_data.unbatch().batch(10).take(1)
for images, labels in validation_batch:
    recostructed_images = model.predict(images, verbose=0) 
    validator.visual_validation(images, recostructed_images, 'visual_validation_test',
                                eval_path)