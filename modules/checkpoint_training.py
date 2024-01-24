import os
import sys
import pandas as pd
from keras.utils import plot_model
from keras.models import save_model
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
from modules.components.data_assets import PreProcessing
from modules.components.model_assets import ModelTraining, RealTimeHistory, ModelValidation, DataGenerator, Inference
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
''')

preprocessor = PreProcessing()

# load the model for inference and print summary
#------------------------------------------------------------------------------
inference = Inference() 
model = inference.load_pretrained_model(GlobVar.model_path)
parameters = inference.model_configuration
model_path = inference.model_path
model.summary(expand_nested=True)

# calculate epochs
#------------------------------------------------------------------------------
initial_epoch = parameters['Epochs']
extra_epochs = initial_epoch + cnf.epochs

# load preprocessed csv files (train and test datasets)
#------------------------------------------------------------------------------
file_loc = os.path.join(model_path, 'preprocessing', 'train_data.csv') 
train_data = pd.read_csv(file_loc, encoding = 'utf-8', sep = (';' or ',' or ' ' or  ':'), low_memory=False)
file_loc = os.path.join(model_path, 'preprocessing', 'test_data.csv') 
test_data = pd.read_csv(file_loc, encoding = 'utf-8', sep = (';' or ',' or ' ' or  ':'), low_memory=False)

# Print report with info about the training parameters
#------------------------------------------------------------------------------
print(f'''
-------------------------------------------------------------------------------
FeXT checkpoint training report
-------------------------------------------------------------------------------
Number of train samples: {train_data.shape[0]}
Number of test samples:  {test_data.shape[0]}
-------------------------------------------------------------------------------
Batch size:              {parameters['Batch size']}
Epochs (pretrained):     {initial_epoch}
Epochs (extra training): {extra_epochs}
-------------------------------------------------------------------------------
''')

# [DEFINE DATA GENERATOR FOR THE IMAGES AND BUILD TF.DATASET]
#==============================================================================
# module for the selection of different operations
#==============================================================================
trainer = ModelTraining(device=cnf.training_device, seed=parameters['Seed'], 
                        use_mixed_precision=cnf.use_mixed_precision)

# define model data generator (train data)
#------------------------------------------------------------------------------
train_generator = DataGenerator(train_data, parameters['Batch size'], parameters['Picture size'], 
                                augmentation=parameters['Augmentation'], shuffle=True)
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
test_generator = DataGenerator(test_data, parameters['Batch size'], parameters['Picture size'], 
                               augmentation=parameters['Augmentation'], shuffle=True)
x_batch, y_batch = test_generator.__getitem__(0)

# create tf.dataset from generator and set prefetch (test data)
#------------------------------------------------------------------------------
output_signature = (tf.TensorSpec(shape=x_batch.shape, dtype=tf.float32), 
                    tf.TensorSpec(shape=y_batch.shape, dtype=tf.float32))
test_dataset = tf.data.Dataset.from_generator(lambda : test_generator, 
                                              output_signature=output_signature)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# [TRAINING WITH FEXT]
#==============================================================================
# Setting callbacks and training routine for the features extraction model. 
# use command prompt on the model folder and (upon activating environment), 
# use the bash command: python -m tensorboard.main --logdir tensorboard/
#==============================================================================

# initialize the real time history callback
#------------------------------------------------------------------------------
RTH_callback = RealTimeHistory(model_path, validation=True)
callbacks_list = [RTH_callback]

# initialize tensorboard if requested
#------------------------------------------------------------------------------
if parameters['Tensorboard'] == True:
    log_path = os.path.join(model_path, 'tensorboard')
    callbacks_list.append(tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1))

# training loop and save model at end of training
#------------------------------------------------------------------------------
training = model.fit(train_dataset, epochs=extra_epochs, validation_data=test_dataset,
                     initial_epoch=initial_epoch, callbacks=callbacks_list, workers=6, 
                     use_multiprocessing=True)

save_model(model, model_path)

# save model parameters in json files
#------------------------------------------------------------------------------
parameters = {'Number of samples' : cnf.num_samples,
              'Picture size' : cnf.pic_size, 
              'Picture shape' : cnf.image_shape,
              'Kernel size' : cnf.kernel_size,             
              'Augmentation' : cnf.augmentation,              
              'Batch size' : cnf.batch_size,
              'Learning rate' : cnf.learning_rate,
              'Epochs' : extra_epochs,
              'Seed' : cnf.seed,
              'Tensorboard' : cnf.use_tensorboard}

trainer.model_parameters(parameters, model_path)

# [FEXT MODEL VALIDATION]
#==============================================================================
# ...
#==============================================================================
validator = ModelValidation(model)

# extract batch of real and reconstructed images and perform visual validation (train set)
#------------------------------------------------------------------------------
val_generator = DataGenerator(train_data, 6, cnf.pic_size, augmentation=False, shuffle=False)
original_images, y_val = val_generator.__getitem__(0)
recostructed_images = list(model.predict(original_images))
validator.FEXT_validation(original_images, recostructed_images, 'train', model_path)

# extract batch of real and reconstructed images and perform visual validation (test set)
#------------------------------------------------------------------------------
val_generator = DataGenerator(test_data, 6, cnf.pic_size, augmentation=False, shuffle=False)
original_images, y_val = val_generator.__getitem__(0)
recostructed_images = list(model.predict(original_images))
validator.FEXT_validation(original_images, recostructed_images, 'test', model_path)

