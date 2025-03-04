# FeXT AutoEncoder: Extraction of Images Features

## 1. Project Overview
The FeXT AutoEncoder project is centered around the development, evaluation, and application of a Convolutional AutoEncoder (CAE) model specifically designed for efficient image feature extraction. The architecture of this model draws inspiration from the renowned VGG16 model, a deep learning framework widely utilized in various computer vision tasks such as image reconstruction, anomaly detection, and feature extraction. As such, the model proposed in this project implements a series of stacked convolution layers, where pooling operations are performed to decrease the encoding dimensions recursively. Both the encoder and the decoder collaboratively work to extract salient features from input images, compressing the information into a lower-dimensional representation suitable for a wide range of downstream tasks.

![VVG16 encoder](FEXT/commons/assets/VGG16_encoder.png)
Architecture of the VVG16 encoder

## 2. FeXT AutoEncoder model
The encoder component of the FeXT AutoEncoder is responsible for image encoding into a lower-dimension latent space. It achieves this through a series of convolutional layers with a kernel size of 2x2 and a stride of 1 pixel, with the downstream convolution layers being followed by max pooling operations. This allows to progressively downsample the spatial dimensions of the input image while expanding the channel dimensions (depth), effectively capturing the abstract representations of the image content. Each stack of convolutional layers is parametrized to use residual connections with layer normalization.

In contrast, the decoder is responsible for reconstructing the original image from the lower-dimensional encoded representation. This is achieved using transposed 2D convolutions and direct upsampling with 2x2 kernels. The decoder aims to restore the spatial dimensions and pixel details of the original image as faithfully as possible by leveraging the abstract features encoded by the model. 

## 3. Training dataset
The FeXT AutoEncoder model has been trained and tested on the Flickr 30K dataset (https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset), a comprehensive collection of images commonly used in many computer vision tasks. The versatility of the FeXT AutoEncoder allows it to be trained on any appropriately preprocessed image dataset, making it adaptable to a wide range of image data and tasks.

## 4. Installation
The installation process on Windows has been designed for simplicity and ease of use. To begin, simply run *start_on_windows.bat.* On its first execution, the installation procedure will automatically start with minimal user input required. The script will check if either Anaconda or Miniconda is installed on your system. If neither is found, it will automatically download and install the latest Miniconda release from https://docs.anaconda.com/miniconda/. After setting up Anaconda/Miniconda, the installation script will proceed with the installation of all necessary Python dependencies. This includes Keras 3 (with PyTorch support as the backend) and the required CUDA dependencies (CUDA 12.4) to enable GPU acceleration. If you'd prefer to handle the installation process separately, you can run the standalone installer by executing *setup/install_on_windows.bat*.

**Important:** After installation, if the project folder is moved or its path is changed, the application will no longer function correctly. To fix this, you can either:

- Open the main menu, select *Setup and maintentance* and choose *Install project in editable mode*
- Manually run the following commands in the terminal, ensuring the project folder is set as the current working directory (CWD):

    `conda activate FEXT`

    `pip install -e . --use-pep517` 

### 4.1 Just-In-Time (JIT) Compiler
This project leverages Just-In-Time model compilation through `torch.compile`, enhancing model performance by tracing the computation graph and applying advanced optimizations like kernel fusion and graph lowering. This approach significantly reduces computation time during both training and inference. The default backend, TorchInductor, is designed to maximize performance on both CPUs and GPUs. Additionally, the installation includes Triton, which generates highly optimized GPU kernels for even faster computation on NVIDIA hardware. For Windows users, a precompiled Triton wheel is bundled with the installation, ensuring seamless integration and performance improvements.

## 5. How to use
On Windows, run *start_on_windows.bat* to launch the main navigation menu and browse through the various options. Please note that some antivirus software, such as Avast, may flag or quarantine python.exe when called by the .bat file. If you encounter unusual behavior, consider adding an exception for your Anaconda or Miniconda environments in your antivirus settings.

**Environmental variables** are stored in *resources/variables/.env*. For security reasons, this file is typically not uploaded to GitHub. Instead, you must create this file manually by copying the template from *resources/templates/.env* and placing it in the *resources/variables* directory.

**KERAS_BACKEND** – Sets the backend for Keras, default is PyTorch.

**TF_CPP_MIN_LOG_LEVEL** – Controls TensorFlow logging verbosity. Setting it to 1 reduces log messages, showing only warnings and errors.

### 5.1 Navigation menu

**1) Analyze image dataset:** perform data validation using a series of metrics for image statistics. 

**2) Model training and evaluation:** open the machine learning menu to explore various options for model training and validation. Once the menu is open, you will see different options:
- **train from scratch:** start training an instance of the autoencoder model from scratch. 
- **train from checkpoint:** resume training from a pretrained checkpoint for an additional amount of epochs, using pretrained model settings and data.  
- **model evaluation:** evaluate the performance of pretrained model checkpoints using different metrics. 

**3) Encode images:** select a model checkpoint and use it to extract abstract representation of image features in the form of lower-dimension embeddings, which will be saved as .npy files. 

**4) Setup and Maintenance:** execute optional commands such as *Install project into environment* to run the developer model project installation, *update project* to pull the last updates from github, and *remove logs* to remove all logs saved in *resources/logs*. 

**5) Exit:** close the program immediately 

### 5.2 Resources
This folder is used to organize data and results for various stages of the project, including data validation, model training, and evaluation. Here are the key subfolders:

- **checkpoints:**  pretrained model checkpoints are stored here, and can be used either for resuming training or performing inference with an already trained model.

- **dataset:** This folder contains images used to train the autoencoder model. Ensure your training data is placed here, and that the images format is of valid type (preferably either .jpg or .png).

- **extraction:**
Contains the subfolder *images* where you place images intended as an input for inference using the pretrained encoder. The resulting lower-dimension projections of the images are saved here as .npy files.

- **logs:** the application logs are saved within this folder

- **validation:** Used to save the results of data validation processes. This helps in keeping track of validation metrics and logs.

## 6. Configurations
For customization, you can modify the main configuration parameters using *settings/configurations.json*. 

#### General Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| SEED               | Global seed for all numerical operations                 |

#### Dataset Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| SAMPLE_SIZE        | Number of samples to use from the dataset                |
| VALIDATION_SIZE    | Proportion of the dataset to use for validation          |
| IMG_AUGMENT        | Whether to apply data augmentation to images             |
| SPLIT_SEED         | Seed for random splitting of the dataset                 |

#### Model Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| JIT_COMPILE        | Apply Just-In_time (JIT) compiler for model optimization |
| JIT_BACKEND        | Just-In_time (JIT) backend                               |

#### Device Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| DEVICE             | Device to use for training (e.g., GPU)                   |
| DEVICE ID          | ID of the device (only used if GPU is selected)          |
| MIXED_PRECISION    | Whether to use mixed precision training                  |
| NUM_PROCESSORS     | Number of processors to use for data loading             |

#### LR Scheduler Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| INITIAL_LR         | Initial value of learni rate                             |
| CONSTANT_STEPS     | Number of steps (batch) to keep the learning rate stable |
| DECAY_STEPS        | Number of steps (batch) to decay learning rate           |

#### Training Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| EPOCHS             | Number of epochs to train the model                      |
| ADDITIONAL EPOCHS  | Number of epochs to train the model from checkpoint      |
| BATCH_SIZE         | Number of samples per batch                              |
| USE_TENSORBOARD    | Whether to use TensorBoard for logging                   |
| SAVE_CHECKPOINTS   | Save checkpoints during training (at each epoch)         |
 
## 7. License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.
