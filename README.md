# FeXT AutoEncoder: Extraction of Images Features

## 1. Project Overview
The FeXT AutoEncoder project is centered around the development, evaluation, and application of a Convolutional AutoEncoder (CAE) model specifically designed for efficient image feature extraction. The architecture of this model draws inspiration from the renowned VGG16 model, a deep learning framework widely utilized in various computer vision tasks such as image reconstruction, anomaly detection, and feature extraction. This model comprises two primary components: the encoder and the decoder. However, the encoder submodel is modified as such that the raw images are passed through a Sobel filter that computes the pixels gradient and join this information with the upstream convolution output. These components collaboratively work to extract salient features from input images, compressing the information into a lower-dimensional vector representation compared to the original image size of 160x160x3 (selected as default input shape, though it could be modified). This compression allows for the retention of critical image information while reducing dimensionality, making the extracted features suitable for a wide range of downstream tasks.

### 1.2 Supplementary information
Further information are available in the `docs` folder (to be added).

## 2. FeXT AutoEncoder model
The encoder component of the FeXT AutoEncoder is responsible for feature extraction. It achieves this through a series of convolutional layers with a kernel size of 3x3 and a stride of 1 pixel. The kernel size is chosen to be compatible with the implementation of the Sobel filter layer, which allows to extract information about the pixel gradients and use them in conjunction with the upstream convoluted tensor, passing the results of the tensor normalized sum to a stack of downstream convolution layers followed by average pooling operations. This allows to progressively downsample the spatial dimensions of the input image while expanding the channel dimensions, effectively capturing the abstract representations of the image content. Each stack of convolutional layers is stabilized with batch normalization and enhanced with ReLU activation functions to introduce non-linearity, enabling the model to learn complex patterns within the data.

In contrast, the decoder component is tasked with reconstructing the original image from the lower-dimensional encoded representation. This is accomplished by reversing the operations performed by the encoder: processing the compressed feature maps using transposed convolutions and direct upsampling with 3x3 kernels. The decoder works to reconstruct the spatial dimensions and pixel details of the original image as accurately as possible from the abstract features encoded by the model.

## 3. Training dataset
The FeXT AutoEncoder model has been trained and tested on the Flickr 30K dataset (https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset), a comprehensive collection of images commonly used in many computer vision tasks. The versatility of the FeXT AutoEncoder allows it to be trained on any appropriately preprocessed image dataset, making it adaptable to a wide range of image data and tasks.

## 4. Installation
The installation process is designed for simplicity, using .bat scripts to automatically create a virtual environment with all necessary dependencies. Please ensure that Anaconda or Miniconda is properly installed on your system before proceeding.

- To set up the environment, run `scripts/environment_setup.bat`. This script installs Keras 3 with pytorch support as backend, and includes includes all required CUDA dependencies to enable GPU utilization (CUDA 12.1).
- **IMPORTANT:** if the path to the project folder is changed for any reason after installation, the app will cease to work. Run `scripts/package_setup.bat` or alternatively use `pip install -e . --use-pep517` from cmd when in the project folder (upon activating the conda environment).

### 3.1 Additional Package for XLA Acceleration
XLA is designed to optimize computations for speed and efficiency, particularly beneficial when working with TensorFlow and other machine learning frameworks that support XLA. Since this project uses Keras 3 with PyTorch as backend, the approach for optimizing computations for speed and efficiency has shifted from XLA to PyTorch's native acceleration tools, particularly TorchScript. This latter allows for the compilation of PyTorch models into an optimized, efficient form that enhances performance, especially when working with large-scale machine learning models or deploying models in production. TorchScript is designed to accelerate both CPU and GPU computations without requiring additional environment variables or complex setup.

For those who wish to use Tensorflow as backend in their own fork of the project, XLA acceleration can be globally enables across your system setting an environment variable named `XLA_FLAGS`. The value of this variable should be `--xla_gpu_cuda_data_dir=path\to\XLA`, where `path\to\XLA` must be replaced with the actual directory path that leads to the folder containing the nvvm subdirectory. It is crucial that this path directs to the location where the file `libdevice.10.bc` resides, as this file is essential for the optimal functioning of XLA. This setup ensures that XLA can efficiently interface with the necessary CUDA components for GPU acceleration.

## 4. How to use
Within the main project folder (FEXT) you will find other folders, each designated to specific tasks. 

### 4.1 Resources
This folder is used to organize data and results for various stages of the project, including data validation, model training, and evaluation. Here are the key subfolders:

**checkpoints:**  pretrained model checkpoints are stored here, and can be used either for resuming training or performing inference with an already trained model.

**dataset:** This folder contains images used to train the autoencoder model. Ensure your training data is placed here, and that the images format is of valid type (preferably either .jpg or .png).

**extraction:**
- `input images:` This subfolder is where you place images intended as an input for inference using the pretrained encoder.
- `image features:` After running the inference script, the resulting lower-dimension embeddings of the input images are saved here as npy files.

**logs:** the application logs are saved within this folder

**validation:** Used to save the results of data validation processes. This helps in keeping track of validation metrics and logs.

### 4.2 Inference
Here you can find the necessary files to run pretrained models in inference mode and use them to extract major features from images

- Run `images_encoding.py` to use the pretrained encoder from a selected model checkpoint to extract abstract representation of image features in the form of lower-dimension embeddings, and save them as npy files. 

### 4.3 Training
This folder contains the necessary files for conducting model training and evaluation: 
- Run `model_training.py` to initiate the FeXT AutoEncoder training process from scratch, initializing a brand new checkpoint
- Run `train_from_checkpoint.py` to resume training from a previous saved checkpoint, using the specific configurations of the model

### 4.4 Validation
Data validation and pretrained model evaluations are performed using the scripts within this folder.
- Launch the jupyter notebook `model_evaluation.ipynb` to evaluate the performance of pretrained model checkpoints using different metrics.
- Launch the jupyter notebook `data_validation.ipynb` to validate the available dataset with different metrics.

## 5. Configurations
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
                    
 
## 6. License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.
