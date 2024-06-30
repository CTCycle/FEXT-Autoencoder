# Advanced settings for training 
#------------------------------------------------------------------------------
MIXED_PRECISION = False
USE_TENSORBOARD = False
XLA_STATE = False # not compatible with current model architecture (default to False)
ML_DEVICE = 'GPU'
NUM_PROCESSORS = 6

# Settings for training routine
#------------------------------------------------------------------------------
EPOCHS = 30
LEARNING_RATE = 0.0001
BATCH_SIZE = 20

# Autoencoder settings
#------------------------------------------------------------------------------
IMG_SHAPE = (256, 256, 3)
SAVE_MODEL_PLOT = True

# Settings for training data 
#------------------------------------------------------------------------------
SAMPLE_SIZE = None
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.1
IMG_AUGMENT = False

# General settings 
#------------------------------------------------------------------------------
SEED = 54
SPLIT_SEED = 45
