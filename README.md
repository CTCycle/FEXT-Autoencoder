# FeXT AutoEncoder: Extraction of Images Features

## 1. Project Overview
This project is dedicated to the training, evaluation, and application of a Convolutional AutoEncoder, specifically designed for image feature extraction. The architecture of this autoencoder is based on the renowned VGG16 model, a deep learning model frequently employed in tasks such as image reconstruction, anomaly detection, and feature extraction. The FEXT AutoEncoder mirrors the structure of the VGG16 model, comprising two primary components (the encoder and the decoder), which are responsabile for extracting relevant features from the images and compressing this information within a vector with lower dimensionality when compared to the original images with size of 256x256x3. The FEXT AutoEncoder model has been trained and tested on the Flickr 30K dataset, typically used for image captioning, and is versatile enough to be trained on any image dataset of your choice, as long as the input have been properly shaped. 

## 2. Applications
In terms of application scope, this autoencoder can be used in a variety of tasks. For example, it can be used for image reconstruction tasks where the goal is to recreate an input image after it has been encoded. They can also be used for anomaly detection tasks where the goal is to identify data points that do not conform to expected behavior. Furthermore, they can be used for feature extraction tasks where the goal is to identify and extract meaningful features from input data. This project is focused on this latter purpose, where the idea is to transform the images in a vectorized representation that holds the most important information.

## 3. FEXT AutoEncoder model

### 3.1 Encoder module
The encoder part of the FEXT autoencoder is composed of convolutional layers that progressively downsample the input image, extracting features at various scales. This part of the network is responsible for transforming the input data into a lower-dimensional representation, namely a 1D vector of size 2048. Input images are resized to shape 256x256x3 to be compatible with the encoder input. The output of the encoder is a compressed features vector of shape 8x8x512.

### 3.2 Decoder module
The decoder part of the VGG16 autoencoder is typically composed of layers that perform the opposite operation to the encoder, upsampling the lower-dimensional representation back to the original input size. This part of the network is responsible for reconstructing the input data from its lower-dimensional representation, obtaining on output with shape 256x256x3.

## 4. Installation
The installation process is designed for simplicity, using .bat scripts to automatically create a virtual environment with all necessary dependencies. Please ensure that Anaconda or Miniconda is installed on your system before proceeding.

- To set up a CPU-only environment, run `setup/create_cpu_environment.bat`. This script installs the base version of TensorFlow, which is lighter and does not include CUDA libraries.
- For GPU support, which is necessary for model training on a GPU, use `setup/create_gpu_environment.bat`. This script includes all required CUDA dependencies to enable GPU utilization. 

### 4.1 Additional Package for XLA Acceleration
XLA is designed to optimize computations for speed and efficiency, particularly beneficial when working with TensorFlow and other machine learning frameworks that support XLA. By incorporating XLA acceleration, you can achieve significant performance improvements in numerical computations, especially for large-scale machine learning models. XLA integration is directly available in TensorFlow but may require enabling specific settings or flags. 

To enable XLA acceleration globally across your system, you need to set an environment variable named `XLA_FLAGS`. The value of this variable should be `--xla_gpu_cuda_data_dir=path\to\XLA`, where `path\to\XLA` must be replaced with the actual directory path that leads to the folder containing the nvvm subdirectory. It is crucial that this path directs to the location where the file `libdevice.10.bc` resides, as this file is essential for the optimal functioning of XLA. This setup ensures that XLA can efficiently interface with the necessary CUDA components for GPU acceleration.

## 5. How to use
The project is organized into subfolders, each dedicated to specific tasks. The `FEXT/utils` folder houses crucial components utilized by various scripts. It's critical to avoid modifying these files, as doing so could compromise the overall integrity and functionality of the program.

**Data:** this folder contains the data utilized for the model training (images are loaded in `FEXT/data/images`). Run the jupyter notebook `FEXT/data/data_validation.ipynb` to conduct an Explorative Data Analysis (EDA) of the image dataset, with the results being saved in `FEXT/data/validation`. 

**Training:** contained within this folder are the necessary files for conducting model training and evaluation. The training model checkpoints are saved in `FEXT AutoEncoder/training/checkpoints`. Run `FEXT/training/model_training.py` to initiate the training process for the autoencoder, or launch the jupyter notebook `FEXT/training/model_evaluation.py` to evaluate the performance of pretrained model checkpoints using different metrics.

**Inference:** run `FEXT/inference/features_extraction.py` to use pretrained model to extract compressed vectorized features from images located within `FEXT/inference/images`. The resulting .csv file is then saved within the same directory.
 
### 5.1 Configurations
For customization, you can modify the main script parameters via the `FEXT/config/configurations.py` file. 

| Category                | Setting                | Description                                                       |
|-------------------------|------------------------|-------------------------------------------------------------------|
| **Advanced settings**   | MIXED_PRECISION        | use mixed precision for faster training (float16/32)              |
|                         | USE_TENSORBOARD        | Activate/deactivate tensorboard logging                           |
|                         | XLA_STATE              | Use linear algebra acceleration for faster training               |
|                         | ML_DEVICE              | Select the training device (CPU or GPU)                           |
|                         | NUM_PROCESSORS         | Number of processors (cores) to use; 1 disables multiprocessing   |
| **Training settings**   | EPOCHS                 | Number of training iterations                                     |
|                         | LEARNING_RATE          | Learning rate of the model                                        |
|                         | BATCH_SIZE             | Size of batches for model training                                |
| **Model settings**      | IMG_SHAPE              | Full shape of the images as (height, width, channels)             |
|                         | SAVE_MODEL_PLOT        | Generate/save 2D model graph (as .png file)                       |
| **Data settings**       | TRAIN_SAMPLES          | Number of images for model training                               |
|                         | TEST_SAMPLES           | Number of samples for validation data                             |
|                         | IMG_AUGMENT            | Perform data augmentation on images (affects training time)       |
| **General settings**    | SEED                   | Global random seed                                                |
|                         | SPLIT_SEED             | Seed for dataset splitting                                        |
                    
 
## License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.
