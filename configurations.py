#------------------------------------------------------------------------------
use_mixed_precision = True
generate_model_graph = True
use_tensorboard = False
XLA_acceleration = False

#------------------------------------------------------------------------------
training_device = 'GPU'
learning_rate = 0.001
batch_size = 25
epochs = 1
kernel_size = 2
seed = 72

#------------------------------------------------------------------------------
num_train_samples = 1000
num_test_samples = 200
picture_shape = (256, 256, 3)
augmentation = False



