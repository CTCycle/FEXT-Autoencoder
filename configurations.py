# Advanced settings for training 
#------------------------------------------------------------------------------
use_mixed_precision = True
use_tensorboard = False
XLA_acceleration = False
training_device = 'GPU'

# Settings for training routine
#------------------------------------------------------------------------------
epochs = 2
learning_rate = 0.0001
batch_size = 20

# Autoencoder settings
#------------------------------------------------------------------------------
picture_shape = (256, 256, 3)
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
