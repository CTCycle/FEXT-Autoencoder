# Advanced settings for training 
#------------------------------------------------------------------------------
use_mixed_precision = False
use_tensorboard = False
XLA_acceleration = False
training_device = 'GPU'

# Settings for training routine
#------------------------------------------------------------------------------
epochs = 2
learning_rate = 0.0001
batch_size = 1

# Autoencoder settings
#------------------------------------------------------------------------------
picture_shape = (64, 64, 3)
kernel_size = 2
generate_model_graph = True

# Settings for training data 
#------------------------------------------------------------------------------
num_train_samples = 500
num_test_samples = 100
augmentation = False

# General settings 
#------------------------------------------------------------------------------
seed = 54
split_seed = 45
