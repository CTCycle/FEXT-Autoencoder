import os
import sys
import pandas as pd
import tensorflow as tf
from keras.models import Model
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# [IMPORT MODULES AND CLASSES]
#==============================================================================
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  
from modules.components.training_classes import ModelTraining, DataGenerator
from modules.components.data_classes import PreProcessing
import modules.global_variables as GlobVar
import modules.configurations as cnf

# [ADD PATH TO XRAY DATASET]
#==============================================================================
# module for the selection of different operations
#==============================================================================
print('''
-------------------------------------------------------------------------------
Features Extraction: extraction from pretrained model
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
dataset = pd.DataFrame(images_paths, columns = ['images path'])

# [DEFINE DATA GENERATOR FOR THE IMAGES AND BUILD TF.DATASET]
#==============================================================================
# module for the selection of different operations
#==============================================================================
trainworker = ModelTraining(device = cnf.training_device, seed = cnf.seed, 
                            use_mixed_precision=cnf.use_mixed_precision)

# define model data generator (train data)
#------------------------------------------------------------------------------
datagenerator = DataGenerator(dataset, cnf.batch_size, cnf.pic_size, 
                                augmentation=False, shuffle=True)
x_batch, y_batch = datagenerator.__getitem__(0)

# create tf.dataset from generator and set prefetch (train data)
#------------------------------------------------------------------------------
output_signature = (tf.TensorSpec(shape=x_batch.shape, dtype=tf.float32), 
                    tf.TensorSpec(shape=y_batch.shape, dtype=tf.float32))
train_dataset = tf.data.Dataset.from_generator(lambda : datagenerator, 
                                               output_signature=output_signature)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# [LOAD PRETRAINED FEXT MODEL]
#==============================================================================
# module for the selection of different operations
#==============================================================================

# define truncated model to get bottleneck layer outputs
#------------------------------------------------------------------------------
trainworker = ModelTraining(device = GlobVar.training_device) 
model = trainworker.load_pretrained_model(GlobVar.model_path)
model.summary(expand_nested=True)

encoder_layer = model.get_layer('encoder_output')
encoder_output = encoder_layer.output
encoder_model = Model(inputs=model.input, outputs=encoder_output)
encoder_model.summary()

# predict compressed image representation
#------------------------------------------------------------------------------
features = []
total_batches = int(dataset.shape[0]/datagenerator.batch_size)
for i, (X, Y) in tqdm(enumerate(datagenerator), total=total_batches):    
    extracted_features = encoder_model.predict(X, verbose = 0)
    flatten_features = [x.reshape(-1) for x in extracted_features]      
    for batch in flatten_features:         
        batch = [str(x) for x in batch]
        string = ' '.join(batch)
        features.append(string)
    if i + 1 == total_batches:
        break

# add predicted vectors to the dataframe
#------------------------------------------------------------------------------
dataset['features'] = features

# [SAVE CSV DATA]
#==============================================================================
# module for the selection of different operations
#==============================================================================
file_loc = os.path.join(GlobVar.data_path, 'images_dataset.csv')  
dataset.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')


