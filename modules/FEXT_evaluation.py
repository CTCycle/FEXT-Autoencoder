import os
import sys
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add modules path if this file is launched as __main__
#------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  

# import modules and components
#------------------------------------------------------------------------------
from modules.components.training_classes import ModelTraining, ModelValidation, DataGenerator
from modules.components.data_classes import PreProcessing
import modules.global_variables as GlobVar
import modules.configurations as cnf

# [ADD PATH TO XRAY DATASET]
#==============================================================================
# module for the selection of different operations
#==============================================================================
print('''
-------------------------------------------------------------------------------
FEXT-AutoEncoder evaluation
-------------------------------------------------------------------------------
.... 
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
df_images = pd.DataFrame(images_paths, columns = ['images path'])
subset_images = df_images.sample(n=cnf.num_samples, random_state=36)

# create test dataset
#------------------------------------------------------------------------------
test_data = subset_images.sample(n=cnf.num_test_samples, random_state=36)
train_data = subset_images.drop(test_data.index)

print(f'''
-------------------------------------------------------------------------------
Number of samples in the dataset = {cnf.num_samples}
Train samples = {cnf.num_samples - cnf.num_test_samples}
Test samples = {cnf.num_test_samples}
Batch size = {cnf.batch_size}
-------------------------------------------------------------------------------
''')

# [DEFINE DATA GENERATOR FOR THE IMAGES AND BUILD TF.DATASET]
#==============================================================================
# module for the selection of different operations
#==============================================================================
trainworker = ModelTraining(device = cnf.training_device, seed = cnf.seed, 
                            use_mixed_precision=cnf.use_mixed_precision)

# define model data generator (train data)
#------------------------------------------------------------------------------
train_generator = DataGenerator(train_data, cnf.batch_size, cnf.pic_size, 
                                augmentation=False, shuffle=True)
x_batch, y_batch = train_generator.__getitem__(0)

# create tf.dataset from generator and set prefetch (train data)
#------------------------------------------------------------------------------
output_signature = (tf.TensorSpec(shape=x_batch.shape, dtype=tf.float32), 
                    tf.TensorSpec(shape=y_batch.shape, dtype=tf.float32))
train_dataset = tf.data.Dataset.from_generator(lambda : train_generator, 
                                               output_signature=output_signature)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# define model data generator (test data)
#------------------------------------------------------------------------------
test_generator = DataGenerator(test_data, cnf.batch_size, cnf.pic_size, 
                               augmentation=False, shuffle=True)
x_batch, y_batch = test_generator.__getitem__(0)

# create tf.dataset from generator and set prefetch (test data)
#------------------------------------------------------------------------------
output_signature = (tf.TensorSpec(shape=x_batch.shape, dtype=tf.float32), 
                    tf.TensorSpec(shape=y_batch.shape, dtype=tf.float32))
test_dataset = tf.data.Dataset.from_generator(lambda : test_generator, 
                                              output_signature=output_signature)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# [LOAD PRETRAINED FEXT MODEL]
#==============================================================================
# ....
#==============================================================================
print('''
-------------------------------------------------------------------------------
Load pretrained model
-------------------------------------------------------------------------------
''')
trainworker = ModelTraining(device = cnf.training_device) 
model = trainworker.load_pretrained_model(GlobVar.model_path)
model.summary(expand_nested=True)

validator = ModelValidation(model)

# extract batch of real and reconstructed images and perform visual validation (train set)
#------------------------------------------------------------------------------
val_generator = DataGenerator(train_data, 6, cnf.pic_size, 
                              augmentation=False, shuffle=False)
original_images, y_val = val_generator.__getitem__(0)
recostructed_images = list(model.predict(original_images))
validator.FEXT_validation(original_images, recostructed_images, 'train', trainworker.model_path)

