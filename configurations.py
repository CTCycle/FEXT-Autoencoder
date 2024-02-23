# Advanced settings for training 
#------------------------------------------------------------------------------
use_mixed_precision = True
use_tensorboard = False
XLA_acceleration = False
training_device = 'GPU'
num_processors = 6

# Settings for training routine
#------------------------------------------------------------------------------
epochs = 25
learning_rate = 0.001
batch_size = 25

# Autoencoder settings
#------------------------------------------------------------------------------
picture_shape = (256, 256, 3)
kernel_size = 2
generate_model_graph = True

# Settings for training data 
#------------------------------------------------------------------------------
num_train_samples = 7000
num_test_samples = 1000
augmentation = False

# General settings 
#------------------------------------------------------------------------------
seed = 72
