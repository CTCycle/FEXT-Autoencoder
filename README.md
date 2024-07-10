# FeXT AutoEncoder: Extraction of Images Features

## 1. Project Overview
This project is dedicated to the training, evaluation, and application of a Convolutional AutoEncoder, specifically designed for image feature extraction. The architecture of this autoencoder is based on the renowned VGG16 model, a deep learning model frequently employed in tasks such as image reconstruction, anomaly detection, and feature extraction. The FEXT AutoEncoder mirrors the structure of the VGG16 model, comprising two primary components (the encoder and the decoder), which are responsabile for extracting relevant features from the images and compressing this information within a vector with lower dimensionality, when compared to the original images with size of 256x256x3. Features extraction is performed by using repeated stack of convolutional layers with a kernel size of 2x2 and single-pixel stride, followed by average pooling operations. Each stack of convolutional layers is also stabilized by batch normalization. While the images are downsampled in terms of height and width, their embedding becomes larger on the channels dimension. 

The FEXT AutoEncoder model has been trained and tested on the Flickr 30K dataset, typically used for image captioning, and is versatile enough to be trained on any image dataset of your choice, as long as the input have been properly shaped. 

## 2. Applications
In terms of application scope, this autoencoder can be used in a variety of tasks. For example, it can be used for image reconstruction tasks where the goal is to recreate an input image after it has been encoded. They can also be used for anomaly detection tasks where the goal is to identify data points that do not conform to expected behavior. Furthermore, they can be used for feature extraction tasks where the goal is to identify and extract meaningful features from input data. This project is focused on this latter purpose, where the idea is to transform the images in a vectorized representation that holds the most important information.

## 3. FEXT AutoEncoder model

### 3.1 Encoder module
The encoder part of the FEXT autoencoder is composed of convolutional layers that progressively downsample the input image, extracting features at various scales. This part of the network is responsible for transforming the input data into a lower-dimensional representation, namely a 3D-vector of size 4x4x512. The convolutional encoder is compatible with input size of 256x256x3, therefor dataset images are rescaled both during training and inference. 

### 3.2 Decoder module
The decoder part of the VGG16 autoencoder is typically composed of layers that perform the opposite operation to the encoder, upsampling the lower-dimensional representation back to the original input size. This part of the network reconstructs the input data from its lower-dimensional representation, obtaining on output with shape 256x256x3 as per the rescaled input images.

## 4. Installation
The installation process is designed for simplicity, using .bat scripts to automatically create a virtual environment with all necessary dependencies. Please ensure that Anaconda or Miniconda is installed on your system before proceeding.

- To set up a CPU-only environment, run `scripts/cpu_environment_setup.bat`. This script installs the base version of TensorFlow, which is lighter and does not include CUDA libraries.
- For GPU support, which is necessary for model training on a GPU, use `scripts/gpu_environment_setup.bat`. This script includes all required CUDA dependencies to enable GPU utilization.
- Once the environment has been created, run `scripts/package_setup.bat` to install the app package locally.
- **IMPORTANT:** run `scripts/package_setup.bat` if the path to the project folder is changed for any reason after installation, or the app won't work! 

### 4.1 Additional Package for XLA Acceleration
XLA is designed to optimize computations for speed and efficiency, particularly beneficial when working with TensorFlow and other machine learning frameworks that support XLA. By incorporating XLA acceleration, you can achieve significant performance improvements in numerical computations, especially for large-scale machine learning models. XLA integration is directly available in TensorFlow but may require enabling specific settings or flags. 

To enable XLA acceleration globally across your system, you need to set an environment variable named `XLA_FLAGS`. The value of this variable should be `--xla_gpu_cuda_data_dir=path\to\XLA`, where `path\to\XLA` must be replaced with the actual directory path that leads to the folder containing the nvvm subdirectory. It is crucial that this path directs to the location where the file `libdevice.10.bc` resides, as this file is essential for the optimal functioning of XLA. This setup ensures that XLA can efficiently interface with the necessary CUDA components for GPU acceleration.

## 5. How to use
The project is organized into subfolders, each dedicated to specific tasks. 

**resources:** includes various subfolders for organizing data and results for data validation and model training and evaluation. The `resources/dataset/` folder contains images used to train the autoencoder model. `resources/encoding/` holds both input images for inference and their corresponding encodings from the pretrained encoder model. The `resources/results/` folder is used to save the results of data validation, while `resources/checkpoints/` contains the pretrained model checkpoints. 

**training:** contained within this folder are the necessary files for conducting model training and evaluation. The training model checkpoints are saved in `resources/checkpoints/`. Run `model_training.py` to initiate the training process for the autoencoder, or launch the jupyter notebook `model_evaluation.py` to evaluate the performance of pretrained model checkpoints using different metrics.

**inference:** run `kmages_features_extraction.py` to use the pretrained encoder from a model checkpoint to extract abstract representation of image features in the form of lower-dimension embeddings. This will load images from `resources/encoding/input_images` and save their corresponding encodings in `resources/encoding/encoder_output` as .npy files. 
 
### 5.1 Configurations
For customization, you can modify the main configuration parameters using `configurations.json` in the root project folder. 

#### Dataset Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| SAMPLE_SIZE        | Number of samples to use from the dataset                |
| VALIDATION_SIZE    | Proportion of the dataset to use for validation          |
| IMG_NORMALIZE      | Whether to normalize image data                          |
| IMG_AUGMENT        | Whether to apply data augmentation to images             |
| SPLIT_SEED         | Seed for random splitting of the dataset                 |

#### Model Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| IMG_SHAPE          | Shape of the input images (height, width, channels)      |
| SAVE_MODEL_PLOT    | Whether to save a plot of the model architecture         |

#### Training Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| EPOCHS             | Number of epochs to train the model                      |
| LEARNING_RATE      | Learning rate for the optimizer                          |
| BATCH_SIZE         | Number of samples per batch                              |
| MIXED_PRECISION    | Whether to use mixed precision training                  |
| USE_TENSORBOARD    | Whether to use TensorBoard for logging                   |
| XLA_STATE          | Whether to enable XLA (Accelerated Linear Algebra)       |
| ML_DEVICE          | Device to use for training (e.g., GPU)                   |
| NUM_PROCESSORS     | Number of processors to use for data loading             |
| PLOT_EPOCH_GAP     | Epochs skipped between each point of the training plot   |

#### Evaluation Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| BATCH_SIZE         | Number of samples per batch during evaluation            |    
                    
 
## License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.
