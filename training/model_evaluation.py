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
from utils.data_assets import DataGenerator, TensorDataSet, PreProcessing
from utils.model_assets import ModelValidation, Inference
import utils.global_paths as globpt
import configurations as cnf

# specify relative paths from global paths and create subfolders
#------------------------------------------------------------------------------
images_path = os.path.join(globpt.data_path, 'images')
cp_path = os.path.join(globpt.train_path, 'checkpoints')
os.mkdir(images_path) if not os.path.exists(images_path) else None
os.mkdir(cp_path) if not os.path.exists(cp_path) else None


# [LOAD MODEL AND DATA]
#==============================================================================
# Load data and models
#==============================================================================

preprocessor = PreProcessing()
inference = Inference(cnf.seed) 

# load the model for inference and print summary
#------------------------------------------------------------------------------
model, parameters = inference.load_pretrained_model(cp_path)
model_path = inference.folder_path
model.summary(expand_nested=True)

# load and reprocess data
#------------------------------------------------------------------------------
filepath = os.path.join(model_path, 'preprocessing', 'train_data.csv')                
train_data = pd.read_csv(filepath, sep=';', encoding='utf-8')
filepath = os.path.join(model_path, 'preprocessing', 'test_data.csv')                
test_data = pd.read_csv(filepath, sep=';', encoding='utf-8')

# regenerate paths
train_data = preprocessor.dataset_from_images(images_path, dataset=train_data)
test_data = preprocessor.dataset_from_images(images_path, dataset=test_data)

# initialize the images generator for the train and test data, and create the 
# tf.dataset according to batch shapes
#------------------------------------------------------------------------------
train_generator = DataGenerator(train_data, 20, cnf.picture_shape, 
                                augmentation=False, shuffle=True)
test_generator = DataGenerator(test_data, 20, cnf.picture_shape, 
                               augmentation=False, shuffle=True)

# initialize the TensorDataSet class with the generator instances
# create the tf.datasets using the previously initialized generators 
datamaker = TensorDataSet()
train_dataset = datamaker.create_tf_dataset(train_generator)
test_dataset = datamaker.create_tf_dataset(test_generator)

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
os.mkdir(eval_path) if not os.path.exists(eval_path) else None

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
    validator.visual_validation(images, recostructed_images, 'visual_validation_train', 
                                eval_path)

# perform visual validation for the test dataset (initialize a validation tf.dataset
# with batch size of 10 images)
#------------------------------------------------------------------------------
validation_batch = test_dataset.unbatch().batch(10).take(1)
for images, labels in validation_batch:
    recostructed_images = model.predict(images, verbose=0) 
    validator.visual_validation(images, recostructed_images, 'visual_validation_test',
                                eval_path)