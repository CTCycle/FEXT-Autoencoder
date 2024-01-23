#------------------------------------------------------------------------------
use_mixed_precision = True
generate_model_graph = True
use_tensorboard = False
XLA_acceleration = False

#------------------------------------------------------------------------------
seed = 42
training_device = 'GPU'
learning_rate = 0.001
batch_size = 50
epochs = 1

#------------------------------------------------------------------------------
kernel_size = 4
pic_size = (224, 224)
num_channels = 3
image_shape = pic_size + (num_channels,)

#------------------------------------------------------------------------------
num_samples = 500
num_test_samples = 200
augmentation = False


