import os
import sys
import pandas as pd
import numpy as np
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
from modules.components.model_assets import ModelTraining, ModelValidation, DataGenerator, Inference
import modules.global_variables as GlobVar
import configurations as cnf

# [LOAD MODEL AND DATA]
#==============================================================================
# module for the selection of different operations
#==============================================================================

# load the model for inference and print summary
#------------------------------------------------------------------------------
inference = Inference() 
model, parameters = inference.load_pretrained_model(GlobVar.model_path)
model_path = inference.model_path
model.summary(expand_nested=True)

# load data
#------------------------------------------------------------------------------
filepath = os.path.join(model_path, 'preprocessing', 'train_data.csv')                
train_data = pd.read_csv(filepath, sep= ';', encoding='utf-8')
filepath = os.path.join(model_path, 'preprocessing', 'test_data.csv')                
test_data = pd.read_csv(filepath, sep= ';', encoding='utf-8')

# [DEFINE DATA GENERATOR FOR THE IMAGES AND BUILD TF.DATASET]
#==============================================================================
# module for the selection of different operations
#==============================================================================
print('''
-------------------------------------------------------------------------------
Model evaluation
-------------------------------------------------------------------------------
.... 
''')

trainer = ModelTraining(device=cnf.training_device, seed = cnf.seed, 
                        use_mixed_precision=cnf.use_mixed_precision)

# initialize the images generator for the train data, get batch at initial index
#------------------------------------------------------------------------------
train_generator = DataGenerator(train_data, 200, cnf.picture_shape, 
                                augmentation=cnf.augmentation, shuffle=True)
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
test_generator = DataGenerator(test_data, 200, cnf.picture_shape, 
                               augmentation=cnf.augmentation, shuffle=True)
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

# predict images from train and test subsets
#------------------------------------------------------------------------------
train_eval = model.evaluate(train_dataset)
test_eval = model.evaluate(test_dataset)

# perform visual validation for the train dataset (initialize a validation tf.dataset
# with batch size of 10 images)
#------------------------------------------------------------------------------
train_dataset = train_dataset.batch(10)
input_images, _ = train_dataset.take(1)
recostructed_images = model.predict(input_images, verbose=0)
validator.visual_validation(input_images, recostructed_images, 'visual_validation_train', eval_path)

# perform visual validation for the test dataset (initialize a validation tf.dataset
# with batch size of 10 images)
#------------------------------------------------------------------------------
test_dataset = test_dataset.batch(10)
input_images, _ = test_dataset.take(1)
recostructed_images = model.predict(input_images, verbose=0) 
validator.visual_validation(input_images, recostructed_images, 'visual_validation_test', eval_path)