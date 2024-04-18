# Advanced settings for training 
#------------------------------------------------------------------------------
use_mixed_precision = True
use_tensorboard = False
XLA_acceleration = False
training_device = 'GPU'
num_processors = 6

# Settings for training routine
#------------------------------------------------------------------------------
epochs = 1
learning_rate = 0.0001
batch_size = 25

# Autoencoder settings
#------------------------------------------------------------------------------
picture_shape = (256, 256, 3)
kernel_size = 2
generate_model_graph = True

# Settings for training data 
#------------------------------------------------------------------------------
num_train_samples = 3000
num_test_samples = 300
augmentation = False

# General settings 
#------------------------------------------------------------------------------
seed = 54
split_seed = 45
