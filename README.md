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
The installation process is designed for simplicity, using .bat scripts to automatically create a virtual environment with all necessary dependencies. Please ensure that Anaconda or Miniconda is properly installed on your system before proceeding.

- To set up the environment, run `scripts/environment_setup.bat`. This script installs Keras 3 with pytorch support as backend, and includes includes all required CUDA dependencies to enable GPU utilization (CUDA 12.1).
- **IMPORTANT:** if the path to the project folder is changed for any reason after installation, the app will cease to work. Run `scripts/package_setup.bat` or alternatively use `pip install -e .` from cmd when in the project folder (upon activating the conda environment).

### 4.1 Additional Package for XLA Acceleration
XLA is designed to optimize computations for speed and efficiency, particularly beneficial when working with TensorFlow and other machine learning frameworks that support XLA. By incorporating XLA acceleration, you can achieve significant performance improvements in numerical computations, especially for large-scale machine learning models. XLA integration is directly available in TensorFlow but may require enabling specific settings or flags. 

To enable XLA acceleration globally across your system, you need to set an environment variable named `XLA_FLAGS`. The value of this variable should be `--xla_gpu_cuda_data_dir=path\to\XLA`, where `path\to\XLA` must be replaced with the actual directory path that leads to the folder containing the nvvm subdirectory. It is crucial that this path directs to the location where the file `libdevice.10.bc` resides, as this file is essential for the optimal functioning of XLA. This setup ensures that XLA can efficiently interface with the necessary CUDA components for GPU acceleration.

## 5. How to use
Within the main project folder (FEXT) you will find other folders, each designated to specific tasks. 

### Resources
This folder is used to organize data and results for various stages of the project, including data validation, model training, and evaluation. Here are the key subfolders:

**dataset:** This folder contains images used to train the autoencoder model. Ensure your training data is placed here, and that the images are saved as either [...].

**encoding:**
- `input_images:` This subfolder is where you place images intended for inference using the pretrained encoder.
- `encoder_output:` After running the inference script, the resulting lower-dimension embeddings of the input images are saved here as .npy files.

**results:** Used to save the results of data validation processes. This helps in keeping track of validation metrics and logs.

**checkpoints:**  pretrained model checkpoints are stored here, and can be used either for resuming training or performing inference with an already trained model.

### Inference
Here you can find the necessary files to run pretrained models in inference mode and use them to extract major features from images

- Run `images_encoding.py` to use the pretrained encoder from a model checkpoint to extract abstract representation of image features in the form of lower-dimension embeddings. 

### Training
This folder contains the necessary files for conducting model training and evaluation: 
- Run `model_training.py` to initiate the training process for the autoencoder

### Validation
Data validation and pretrained model evaluations are performed using the scripts within this folder.
- Launch the jupyter notebook `model_evaluation.ipynb` to evaluate the performance of pretrained model checkpoints using different metrics.
- Launch the jupyter notebook `data_validation.ipynb` to validate the available data with different metrics.


### 5.1 Configurations
For customization, you can modify the main configuration parameters using `settings/configurations.json` 

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
