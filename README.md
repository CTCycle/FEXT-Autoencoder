# FEXT AutoEncoder

## Project description
This project is dedicated to the training, evaluation, and application of a Convolutional AutoEncoder, specifically designed for image feature extraction (hence, FEXT). The architecture of this autoencoder is based on the renowned VGG16 model, a neural network frequently employed in deep learning tasks such as image reconstruction, anomaly detection, and feature extraction. The FEXT AutoEncoder mirrors the structure of the VGG16 model, comprising two primary components: the encoder and the decoder. The idea behind the encoder is to return a compressed 1D vector with size 2048, which can be used by the decoder to reconstruct the image if necessary.

The model has been trained and tested on the Flickr 30K dataset, typically used for image captioning. However, it is versatile enough to be trained on any image dataset of your choice, which should be located in the ‘dataset/images’ folder. The architecture of the model and the training routine are defined using Tensorflow (version 2.10). A custom generator is used to feed the images, enabling the handling of large image datasets that might not fit into memory.

## Model structure

### Encoder module
The encoder part of the FEXT autoencoder is composed of convolutional layers that progressively downsample the input image, extracting features at various scales. This part of the network is responsible for transforming the input data into a lower-dimensional representation, namely a 1D vector of size 2048. Input images are resized to shape 256x256 (3 channels) to be compatible with the encoder input.

### Decoder module
The decoder part of the VGG16 autoencoder is typically composed of layers that perform the opposite operation to the encoder, upsampling the lower-dimensional representation back to the original input size. This part of the network is responsible for reconstructing the input data from its lower-dimensional representation, obtaining on output with shape 256x256x3.

## Applications
In terms of application scope, this autoencoder can be used in a variety of tasks. For example, it can be used for image reconstruction tasks where the goal is to recreate an input image after it has been encoded. They can also be used for anomaly detection tasks where the goal is to identify data points that do not conform to expected behavior. Furthermore, they can be used for feature extraction tasks where the goal is to identify and extract meaningful features from input data. This project is focused on this latter purpose, where the idea is to transform the images in a vectorized representation that holds the most important information.

## How to use
Run the FeXTAutoEncoder.py file to launch the script and use the main menu to navigate the different options. From the main menu, you can select one of the following options:

**1) Data validation** Evaluate images dataset from `/images` and compare train and test datasets properties. Data validation is based on training parameters from `configurations.py`.

**2) Pretrain FeXTAutoEncoder model** Pretrain the autoencoder from scratch using the data from `/images`. Data preprocessing is performed prior to start the training session. 

**3) Evaluate FeXTAutoEncoder model** Select a pretrained model and evaluate its performance throguh different methodologies, leveraging both training and validation datasets

**4) Extract features from images** Use the pretrained model in inference mode, to extract features from images and save the compressed vector locally as .csv file

**5) Exit and close**

### Configurations
For customization, you can modify the main script parameters via the ‘configurations.py’ file. The following parameters are available:

**Advanced settings for training:**
- `use_mixed_precision:` whether or not to use mixed precision for faster training (mix float16/float32)
- `use_tensorboard:` activate or deactivate tensorboard logging
- `XLA_acceleration:` use of linear algebra acceleration for faster training 
- `training_device:` select the training device (CPU or GPU)

**Settings for training routine:**
- `epochs:` number of training iterations
- `learning_rate:` learning rate of the model 
- `batch_size:` size of batches to be fed to the model during training

**Autoencoder settings:**
- `picture_shape:` full shape of the images as (height, width, channels)
- `kernel_size:` size of convolutional kernel (best to keep at 2)
- `generate_model_graph:` generate and save 2D model graph (as .png file)

**Settings for training data:**
- `num_train_samples:` number of images to use for the model training 
- `num_test_samples:` number of samples to use as validation data
- `augmentation:` whether or not to perform data agumentation on images (significant impact on training time)

**General settings:**
- `seed:` global random seed

**Number of samples and batch size:** This application is designed for efficient on-the-fly image loading using a custom generator. Therefore, it is advisable to carefully choose the number of training and testing samples and adjust the batch size accordingly. Optimal training and test data sizes should be multiples of 100.
              
## Installation 
First, ensure that you have Python 3.10.12 installed on your system. Then, you can easily install the required Python packages using the provided requirements.txt file:

`pip install -r requirements.txt` 

In addition to the Python packages, certain extra dependencies may be required for specific functionalities. These dependencies can be installed using conda or other external installation methods, depending on your operating system. Specifically, you will need to install graphviz and pydot to enable the visualization of the 2D model architecture:
- graphviz version 2.38.0
- pydot version 1.4.2

You can install these dependencies using the appropriate package manager for your system. For instance, you might use conda or an external installation method based on your operating system's requirements.

## CUDA GPU Support (Optional, for GPU Acceleration)
If you have an NVIDIA GPU and want to harness the power of GPU acceleration using CUDA, please follow these additional steps. The application is built using TensorFlow 2.10.0 to ensure native Windows GPU support, so remember to install the appropriate versions:

### 1. Install NVIDIA CUDA Toolkit (Version 11.2)

To enable GPU acceleration, you'll need to install the NVIDIA CUDA Toolkit. Visit the [NVIDIA CUDA Toolkit download page](https://developer.nvidia.com/cuda-downloads) and select the version that matches your GPU and operating system. Follow the installation instructions provided. Alternatively, you can install `cuda-toolkit` as a package within your environment.

### 2. Install cuDNN (NVIDIA Deep Neural Network Library, Version 8.1.0.77)

Next, you'll need to install cuDNN, which is the NVIDIA Deep Neural Network Library. Visit the [cuDNN download page](https://developer.nvidia.com/cudnn) and download the cuDNN library version that corresponds to your CUDA version (in this case, version 8.1.0.77). Follow the installation instructions provided.

### 3. Additional Package (If CUDA Toolkit Is Installed)

If you've installed the NVIDIA CUDA Toolkit within your environment, you may also need to install an additional package called `cuda-nvcc` (Version 12.3.107). This package provides the CUDA compiler and tools necessary for building CUDA-enabled applications.

### 4. Additional Package for XLA Acceleration

XLA is designed to optimize computations for speed and efficiency, particularly beneficial when working with TensorFlow and other machine learning frameworks that support XLA. By incorporating XLA acceleration, you can achieve significant performance improvements in numerical computations, especially for large-scale machine learning models. XLA integration is directly available in TensorFlow but may require enabling specific settings or flags.

To enable XLA acceleration globally across your system, you need to set an environment variable named `XLA_FLAGS`. The value of this variable should be `--xla_gpu_cuda_data_dir=path\to\XLA`, where `path\to\XLA` must be replaced with the actual directory path that leads to the folder containing the nvvm subdirectory. It is crucial that this path directs to the location where the file `libdevice.10.bc` resides, as this file is essential for the optimal functioning of XLA. This setup ensures that XLA can efficiently interface with the necessary CUDA components for GPU acceleration.

## License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.
