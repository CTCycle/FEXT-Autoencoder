#------------------------------------------------------------------------------
use_mixed_precision = True
generate_model_graph = True
use_tensorboard = False
XLA_acceleration = False

#------------------------------------------------------------------------------
training_device = 'GPU'
learning_rate = 0.001
batch_size = 50
epochs = 20
kernel_size = 2
seed = 56

#------------------------------------------------------------------------------
num_train_samples = 6000
num_test_samples = 1000
picture_shape = (224, 224, 3)
augmentation = False



