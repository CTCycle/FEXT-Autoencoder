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
from modules.components.model_assets import ModelTraining, ModelValidation, DataGenerator
from modules.components.data_assets import PreProcessing
import modules.global_variables as GlobVar
import configurations as cnf

