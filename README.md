# FEXT AutoEncoder

## Project description
An autoencoder for image features representation (FEXT stands for Features EXTraction), based on the VV16G autoencoder architecture. The VGG16 autoencoder is a type of neural network architecture that is often used in the field of deep learning for tasks such as image reconstruction, anomaly detection, and feature extraction. The layout of the VGG16 autoencoder can be divided into two main parts: the encoder and the decoder. As such, the FEXT AutoEncoder proposed in this project is built following the same assumption. The model has been trained using the Flickr 30K dataset (usually used for image captioning), which are loaded in the images folder within dataset (dataset/images), but you can use virtually any set of images. Both the model architecture and training routine are defined using Tensorflow (2.10), with the images being fed using a custom generator (allows for using large datasets of images that may not fit in memory). You can change the main script parameters using the configurations.py file.

## Model structure

### Encoder module
The encoder part of the FEXT autoencoder is composed of convolutional layers that progressively downsample the input image, extracting features at various scales. This part of the network is responsible for transforming the input data into a lower-dimensional representation.

### Decoder module
The decoder part of the VGG16 autoencoder is typically composed of layers that perform the opposite operation to the encoder, upsampling the lower-dimensional representation back to the original input size. This part of the network is responsible for reconstructing the input data from its lower-dimensional representation.

## Applications
In terms of application scope, this autoencoder can be used in a variety of tasks. For example, it can be used for image reconstruction tasks where the goal is to recreate an input image after it has been encoded. They can also be used for anomaly detection tasks where the goal is to identify data points that do not conform to expected behavior. Furthermore, they can be used for feature extraction tasks where the goal is to identify and extract meaningful features from input data. This project is focused on this latter purpose, where the idea is to transform the images in a vectorized representation that holds the most important information.

## How to use
This application is composed of various modules, each performing distinct operations. Run the NISTADS_main.py file to launch the script and use the main menu to navigate the different options.

**The main options are as following:**
1) ...                   
2) ....                                   
3) Exit and close

**Pretrain FEXT-AutoEncoder model**: train the autoencoder on a subset of images from **/images**. The training parameters can be changed using the configurations.py. When the training session is over, the model is saved in the designated folder, together with a json file reporting the model configuration and the outcome of preliminary performance evalutations. 

**Evaluate FEXT-AutoEncoder model**: this module evaluates any selected model (loaded from the **/models**) by analysing performance on train and test sets and overall accuracy of the trained model. 

**Extract features from images**: select a pretrained model to extract vectors of image features (from **/predictions**) and save them in a .csv file

**Exit and close**
                 
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

By following these steps, you can ensure that your environment is configured to take full advantage of GPU acceleration for enhanced performance. 

## License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.
