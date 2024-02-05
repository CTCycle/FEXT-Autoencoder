#------------------------------------------------------------------------------
use_mixed_precision = True
generate_model_graph = True
use_tensorboard = False
XLA_acceleration = False

#------------------------------------------------------------------------------
training_device = 'GPU'
learning_rate = 0.0001
batch_size = 50
epochs = 10
kernel_size = 4
seed = 42

#------------------------------------------------------------------------------
num_train_samples = 1000
num_test_samples = 200
augmentation = False
picture_shape = (224, 224, 3)


