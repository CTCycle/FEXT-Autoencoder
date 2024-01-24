#------------------------------------------------------------------------------
use_mixed_precision = True
generate_model_graph = True
use_tensorboard = False
XLA_acceleration = False

#------------------------------------------------------------------------------
seed = 42
training_device = 'GPU'
learning_rate = 0.0001
batch_size = 25
epochs = 1

#------------------------------------------------------------------------------
kernel_size = 4
pic_size = (224, 224)
image_shape = (224, 224, 3)

#------------------------------------------------------------------------------
num_samples = 1000
num_test_samples = 400
augmentation = False


