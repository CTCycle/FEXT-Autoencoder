#------------------------------------------------------------------------------
use_mixed_precision = True
generate_model_graph = True
use_tensorboard = False
XLA_acceleration = False

#------------------------------------------------------------------------------
training_device = 'GPU'
learning_rate = 0.001
batch_size = 50
epochs = 25
kernel_size = 3
seed = 56

#------------------------------------------------------------------------------
num_train_samples = 5000
num_test_samples = 1000
picture_shape = (208, 208, 3)
augmentation = False



