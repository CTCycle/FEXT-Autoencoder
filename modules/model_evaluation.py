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
from modules.components.model_assets import DataGenerator, ModelValidation, Inference
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
filepath = os.path.join(model_path, 'preprocessing', 'train_data.csv')                
train_data = pd.read_csv(filepath, sep=';', encoding='utf-8')
filepath = os.path.join(model_path, 'preprocessing', 'test_data.csv')                
test_data = pd.read_csv(filepath, sep=';', encoding='utf-8')

# initialize the images generator for the train data, get batch at initial index
#------------------------------------------------------------------------------
train_generator = DataGenerator(train_data, 20, cnf.picture_shape, 
                                augmentation=False, shuffle=True)
x_batch, y_batch = train_generator.__getitem__(0)

# create train tf.dataset from generator and set prefetch scheduler 
#------------------------------------------------------------------------------
output_signature = (tf.TensorSpec(shape=x_batch.shape, dtype=tf.float32), 
                    tf.TensorSpec(shape=y_batch.shape, dtype=tf.float32))
train_dataset = tf.data.Dataset.from_generator(lambda : train_generator, 
                                               output_signature=output_signature)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# initialize the images generator for the test data, get batch at initial index
#------------------------------------------------------------------------------
test_generator = DataGenerator(test_data, 20, cnf.picture_shape, 
                               augmentation=False, shuffle=True)
x_batch, y_batch = test_generator.__getitem__(0)

# create test tf.dataset from generator and set prefetch scheduler 
#------------------------------------------------------------------------------
output_signature = (tf.TensorSpec(shape=x_batch.shape, dtype=tf.float32), 
                    tf.TensorSpec(shape=y_batch.shape, dtype=tf.float32))
test_dataset = tf.data.Dataset.from_generator(lambda : test_generator, 
                                              output_signature=output_signature)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

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

# evluate the model on both the train and test dataset
#------------------------------------------------------------------------------
train_eval = model.evaluate(train_dataset, batch_size=20, verbose=1)
test_eval = model.evaluate(test_dataset, batch_size=20, verbose=1)

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
validation_batch = train_dataset.unbatch().batch(10).take(1)
for images, labels in validation_batch:
    recostructed_images = model.predict(images, verbose=0)
    validator.visual_validation(images, recostructed_images, 
                                'visual_validation_train', 
                                eval_path)

# perform visual validation for the test dataset (initialize a validation tf.dataset
# with batch size of 10 images)
#------------------------------------------------------------------------------
validation_batch = test_dataset.unbatch().batch(10).take(1)
for images, labels in validation_batch:
    recostructed_images = model.predict(images, verbose=0) 
    validator.visual_validation(images, recostructed_images, 
                                'visual_validation_test',
                                eval_path)