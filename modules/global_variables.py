import os

#------------------------------------------------------------------------------
images_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models') 

#------------------------------------------------------------------------------
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(images_path):
    os.mkdir(images_path)

#------------------------------------------------------------------------------
seed = 42
training_device = 'GPU'
learning_rate = 0.0001
pic_size = (224, 224)
batch_size = 30
epochs = 200

#------------------------------------------------------------------------------
num_samples = 5000
num_test_samples = 2000


