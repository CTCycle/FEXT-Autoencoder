import os
import sys
import pandas as pd
from keras.utils import plot_model
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
from modules.components.data_classes import PreProcessing
from modules.components.training_classes import ModelTraining, RealTimeHistory, AutoEncoderModel, ModelValidation, DataGenerator
import modules.global_variables as GlobVar
import configurations as cnf

# [LOAD DATA AND ADD IMAGES PATHS TO DATASET]
#==============================================================================
# Load the csv with data and transform the tokenized text column to convert the
# strings into a series of integers
#==============================================================================
print('''
-------------------------------------------------------------------------------
FEXT-AutoEncoder training
-------------------------------------------------------------------------------
Add description
      
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

# [BUILD FEATURES EXTRACTION MODEL]
#==============================================================================
# ....
#==============================================================================
modelworker = AutoEncoderModel(cnf.learning_rate, cnf.pic_size, XLA_state=cnf.XLA_acceleration)
model = modelworker.FEXT_AutoEncoder() 
model.summary(expand_nested=True)

# generate graphviz plot fo the model layout
#------------------------------------------------------------------------------
model_savepath = preprocessor.model_savefolder(GlobVar.model_path, 'FEXT')
if cnf.generate_model_graph == True:
    plot_path = os.path.join(model_savepath, 'model_layout.png')       
    plot_model(model, to_file = plot_path, show_shapes = True, 
               show_layer_names = True, show_layer_activations = True, 
               expand_nested = True, rankdir = 'TB', dpi = 400)

# [TRAINING WITH FEXT]
#==============================================================================
# Setting callbacks and training routine for the features extraction model. 
# use command prompt on the model folder and (upon activating environment), 
# use the bash command: python -m tensorboard.main --logdir tensorboard/
#==============================================================================

# initialize the real time history callback
#------------------------------------------------------------------------------
RTH_callback = RealTimeHistory(model_savepath, validation=True)

# training loop (with or without tensorboard callback), saves the model at the end of
# the training
#------------------------------------------------------------------------------
print(f'''Start model training for {cnf.epochs} epochs and batch size of {cnf.batch_size}
       ''')
if cnf.use_tensorboard == True:
    log_path = os.path.join(model_savepath, 'tensorboard')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, 
                                                          histogram_freq=1)
    training = model.fit(train_dataset, epochs = cnf.epochs,
                         validation_data=test_dataset, 
                         callbacks = [tensorboard_callback, RTH_callback],
                         workers = 6, use_multiprocessing=True) 
else:
    training = model.fit(train_dataset, epochs = cnf.epochs,
                         validation_data=test_dataset, 
                         callbacks = [RTH_callback],
                         workers = 6, use_multiprocessing=True) 

model.save(model_savepath)

# save model parameters in txt files
#------------------------------------------------------------------------------
parameters = {'Number of samples' : cnf.num_samples,
              'Picture size' : cnf.pic_size,              
              'Batch size' : cnf.batch_size,
              'Learning rate' : cnf.learning_rate,
              'Epochs' : cnf.epochs}

trainworker.model_parameters(parameters, model_savepath)

# [FEXT MODEL VALIDATION]
#==============================================================================
# ...
#==============================================================================
validator = ModelValidation(model)

# extract batch of real and reconstructed images and perform visual validation (train set)
#------------------------------------------------------------------------------
val_generator = DataGenerator(train_data, 6, cnf.pic_size, 
                              augmentation=False, shuffle=False)
original_images, y_val = val_generator.__getitem__(0)
recostructed_images = list(model.predict(original_images))
validator.FEXT_validation(original_images, recostructed_images, 'train', model_savepath)

# extract batch of real and reconstructed images and perform visual validation (test set)
#------------------------------------------------------------------------------
val_generator = DataGenerator(test_data, 6, cnf.pic_size, 
                              augmentation=False, shuffle=False)
original_images, y_val = val_generator.__getitem__(0)
recostructed_images = list(model.predict(original_images))
validator.FEXT_validation(original_images, recostructed_images, 'test', model_savepath)

