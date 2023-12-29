#------------------------------------------------------------------------------
use_tensorboard = False
generate_model_graph = True
XLA_acceleration = True
use_mixed_precision = True

#------------------------------------------------------------------------------
seed = 42
training_device = 'GPU'
learning_rate = 0.001
pic_size = (224, 224)
num_channels = 3
image_shape = pic_size + (num_channels,)
batch_size = 25
epochs = 100

#------------------------------------------------------------------------------
num_samples = 4000
num_test_samples = 800



