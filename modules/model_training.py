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
from modules.components.data_assets import PreProcessing
from modules.components.model_assets import ModelTraining, RealTimeHistory, FeXTAutoEncoder, DataGenerator, LRScheduler
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

# create test dataset
#------------------------------------------------------------------------------
test_data = df_images.sample(n=cnf.num_test_samples, random_state=36)
train_data = df_images.drop(test_data.index)

# create model folder and preprocessing subfolder
#------------------------------------------------------------------------------
model_savepath = preprocessor.model_savefolder(GlobVar.model_path, 'FeXT')
pp_path = os.path.join(model_savepath, 'preprocessing')
if not os.path.exists(pp_path):
    os.mkdir(pp_path)

# save preprocessed data
#------------------------------------------------------------------------------
file_loc = os.path.join(pp_path, 'train_data.csv')  
train_data.to_csv(file_loc, index=False, sep=';', encoding='utf-8')
file_loc = os.path.join(pp_path, 'test_data.csv')  
test_data.to_csv(file_loc, index=False, sep=';', encoding='utf-8')

# [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
#==============================================================================
# ...
#==============================================================================

# initialize training device (allows chaning device prior to initializing the generators)
#------------------------------------------------------------------------------
trainer = ModelTraining(device=cnf.training_device, seed=cnf.seed, 
                        use_mixed_precision=cnf.use_mixed_precision)

# initialize the images generator for the train data, get batch at initial index
#------------------------------------------------------------------------------
train_generator = DataGenerator(train_data, cnf.batch_size, cnf.picture_shape, 
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
test_generator = DataGenerator(test_data, cnf.batch_size, cnf.picture_shape, 
                               augmentation=cnf.augmentation, shuffle=True)
x_batch, y_batch = test_generator.__getitem__(0)

# create test tf.dataset from generator and set prefetch scheduler 
#------------------------------------------------------------------------------
output_signature = (tf.TensorSpec(shape=x_batch.shape, dtype=tf.float32), 
                    tf.TensorSpec(shape=y_batch.shape, dtype=tf.float32))
test_dataset = tf.data.Dataset.from_generator(lambda : test_generator, 
                                              output_signature=output_signature)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

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
FeXT training report
-------------------------------------------------------------------------------
Number of train samples: {train_data.shape[0]}
Number of test samples:  {test_data.shape[0]}
-------------------------------------------------------------------------------
Picture shape:           {cnf.picture_shape}
Kernel size:             {cnf.kernel_size}
Batch size:              {cnf.batch_size}
Epochs:                  {cnf.epochs}
-------------------------------------------------------------------------------
''')

# initialize the learning rate scheduler
#------------------------------------------------------------------------------
warmup_steps = cnf.epochs//10
decay_steps = cnf.epochs//5
decay_rate = 0.1
LR_scheduler = LRScheduler(cnf.learning_rate, decay_steps, decay_rate, warmup_steps)

# build the autoencoder model 
#------------------------------------------------------------------------------
modelworker = FeXTAutoEncoder(cnf.learning_rate, cnf.kernel_size, cnf.picture_shape, 
                              cnf.seed, XLA_state=cnf.XLA_acceleration)
model = modelworker.get_model(summary=True) 

# generate graphviz plot fo the model layout
#------------------------------------------------------------------------------
if cnf.generate_model_graph==True:
    plot_path = os.path.join(model_savepath, 'model_layout.png')       
    plot_model(model, to_file = plot_path, show_shapes = True, 
               show_layer_names = True, show_layer_activations = True, 
               expand_nested = True, rankdir = 'TB', dpi = 400)

# initialize the real time history callback
#------------------------------------------------------------------------------
RTH_callback = RealTimeHistory(model_savepath, validation=True)
callbacks_list = [RTH_callback]

# initialize tensorboard if requested
#------------------------------------------------------------------------------
if cnf.use_tensorboard:
    log_path = os.path.join(model_savepath, 'tensorboard')
    callbacks_list.append(tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1))

# training loop and save model at end of training
#------------------------------------------------------------------------------
training = model.fit(train_dataset, epochs=cnf.epochs, validation_data=test_dataset, 
                     callbacks=callbacks_list, workers=6, use_multiprocessing=True)

model.save(model_savepath, save_format='tf')

# save model parameters in json files
#------------------------------------------------------------------------------
parameters = {'Train_samples': cnf.num_train_samples,
              'Test_samples': cnf.num_test_samples,
              'Picture_shape' : cnf.picture_shape,             
              'Kernel_size' : cnf.kernel_size,              
              'Augmentation' : cnf.augmentation,              
              'Batch_size' : cnf.batch_size,
              'Learning_rate' : cnf.learning_rate,
              'Epochs' : cnf.epochs,
              'Seed' : cnf.seed,
              'Tensorboard' : cnf.use_tensorboard}

trainer.model_parameters(parameters, model_savepath)



