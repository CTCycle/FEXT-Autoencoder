# Advanced settings for training 
#------------------------------------------------------------------------------
MIXED_PRECISION = True
USE_TENSORBOARD = False
XLA_STATE = False
ML_DEVICE = 'GPU'
NUM_PROCESSORS = 6

# Settings for training routine
#------------------------------------------------------------------------------
epochs = 1
learning_rate = 0.0001
batch_size = 40

# Autoencoder settings
#------------------------------------------------------------------------------
picture_shape = (256, 256, 3)
kernel_size = 2
generate_model_graph = True

# Settings for training data 
#------------------------------------------------------------------------------
num_train_samples = 2000
num_test_samples = 200
augmentation = False

# General settings 
#------------------------------------------------------------------------------
seed = 54
split_seed = 45
