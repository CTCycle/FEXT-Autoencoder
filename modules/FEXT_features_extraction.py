import os
import sys
import pandas as pd
from keras.models import Model, load_model
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# [IMPORT MODULES AND CLASSES]
#==============================================================================
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  
from modules.components.training_classes import ModelTraining
from modules.components.data_classes import PreProcessing
import modules.global_variables as GlobVar

# [LOAD TEXT DATA]
#==============================================================================
# module for the selection of different operations
#==============================================================================
file_loc = os.path.join(GlobVar.data_path, 'images_dataset.csv') 
dataset = pd.read_csv(file_loc, encoding = 'utf-8', sep = (';' or ',' or ' ' or  ':'), low_memory = False)

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
dataset = preprocessor.images_pathfinder(GlobVar.images_path, dataset, 'id')
dataset = dataset.sample(n=GlobVar.num_samples)

# [LOAD PRETRAINED FEXT MODEL]
#==============================================================================
# module for the selection of different operations
#==============================================================================
print('''
STEP 1 ----> Features extraction using pretrained FEXT
''')

# define generator to feed prediction loop
#------------------------------------------------------------------------------
trainworker = ModelTraining(device = 'GPU')
generator = trainworker.FEXT_generator(dataset, 'images_path', GlobVar.pic_size[:-1],
                                       GlobVar.batch_size, transform=False, shuffle=False)


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
total_batches = int(dataset.shape[0]/generator.batch_size)
for i, (X, Y) in tqdm(enumerate(generator), total=total_batches):    
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


