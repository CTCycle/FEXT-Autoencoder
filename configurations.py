# Advanced settings for training 
#------------------------------------------------------------------------------
use_mixed_precision = True
use_tensorboard = False
XLA_acceleration = False
training_device = 'GPU'

# Settings for training routine
#------------------------------------------------------------------------------
epochs = 500
learning_rate = 0.001
batch_size = 1024

# Autoencoder settings
#------------------------------------------------------------------------------
picture_shape = (256, 256, 3)
kernel_size = 2
generate_model_graph = True

# Settings for training data 
#------------------------------------------------------------------------------
num_train_samples = 1000
num_test_samples = 200
augmentation = False

# General settings 
#------------------------------------------------------------------------------
seed = 72
