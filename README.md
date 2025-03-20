# FeXT AutoEncoder: Extraction of Images Features

## 1. Project Overview
FeXT AutoEncoder is a project centered around the implementation, training and evaluation of a Convolutional AutoEncoder (CAE) model specifically designed for efficient image feature extraction. The architecture of this model draws inspiration from the renowned VGG16 model, a deep learning framework widely utilized in various computer vision tasks such as image reconstruction, anomaly detection, and feature extraction (https://keras.io/api/applications/vgg/). Hence, the FEXT model implements a stack of convolutional layers, where pooling operations are performed to decrease the spatial dimensions multiple times. Both the encoder and the decoder collaboratively work to extract the most representative features from input images, projecting the original information into a lower-dimensional latent space that could be used for a wide range of downstream tasks.

![VGG16 encoder](FEXT/commons/assets/VGG16_encoder.png)
Architecture of the VGG16 encoder

## 2. FeXT AutoEncoder model
As briefly explained, the encoder component of the FeXT AutoEncoder is responsible for image encoding into a lower-dimension latent space. It achieves this through a series of convolutional layers with a kernel size of 2x2 and a single-pixel stride, being followed by max pooling. This allows to progressively downsample the spatial dimensions of the input image while expanding the channel dimensions (depth), effectively capturing the abstract representations of the image content. Each stack of convolutional layers is parametrized to use residual connections with layer normalization, in order to mitigate issues related to vanishing gradient in deep networks.

In contrast, the decoder is responsible for reconstructing the original image from the lower-dimensional latent space. This is achieved using transposed 2D convolutions and direct nearest-pixel upsampling with 2x2 kernels. The scope of the decoder is to faithfully restore the original image by reconstructing details and pixel distribution by solely using the compressed image projections as a reference.

## 3. Training dataset
The FeXT AutoEncoder model has been trained and tested on the Flickr 30K dataset (https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset), a comprehensive collection of images commonly used in many computer vision tasks. However, this model can be trained on virtually any image dataset, as the inputs will be automatically resized and normalized.

## 4. Installation
The installation process on Windows has been designed to be fully automated. To begin, simply run *start_on_windows.bat.* On its first execution, the installation procedure will execute with minimal user input required. The script will check if either Anaconda or Miniconda is installed and can be accessed from your system path. If neither is found, it will automatically download and install the latest Miniconda release from https://docs.anaconda.com/miniconda/. Following this step, the script will proceed with the installation of all necessary Python dependencies. 

This includes Keras 3 (with PyTorch support as the backend) and the required CUDA dependencies (CUDA 12.4) to enable GPU acceleration. Should you prefer to handle the installation process separately, you can run the standalone installer by running *setup/install_on_windows.bat*.

**Important:** After installation, if the project folder is moved or its path is changed, the application will no longer function correctly. To fix this, you can either:

- Open the main menu, select *Setup and maintentance* and choose *Install project in editable mode*
- Manually run the following commands in the terminal, ensuring the project folder is set as the current working directory (CWD):

    `conda activate FEXT`

    `pip install -e . --use-pep517` 

### 4.1 Just-In-Time (JIT) Compiler
This project leverages Just-In-Time model compilation through `torch.compile`, enhancing model performance by tracing the computation graph and applying advanced optimizations like kernel fusion and graph lowering. This approach significantly reduces computation time during both training and inference. The default backend, TorchInductor, is designed to maximize performance on both CPUs and GPUs. Additionally, the installation includes Triton, which generates highly optimized GPU kernels for even faster computation on NVIDIA hardware. For Windows users, a precompiled Triton wheel is bundled with the installation, ensuring seamless integration and performance improvements.

## 5. How to use
On Windows, run *start_on_windows.bat* to launch the main navigation menu and browse through the various options. Please note that some antivirus software, such as Avast, may flag or quarantine python.exe when called by the .bat file. If you encounter unusual behavior, consider adding an exception for your Anaconda or Miniconda environments in your antivirus settings.

### 5.1 Navigation menu

**1) Analyze image dataset:** analyze and validate the image dataset using different metrics. At first, a summary of images statistics is generated and saved in the image statistics table of the database. This summary includes mean pixel values, pixel standard deviation, pixel values range and noise ratio and standard deviation. Then, the average pixel distribution is calculated and saved into *resources/database/validation*.  

**2) Model training and evaluation:** open the machine learning menu to explore various options for model training and validation.

- **train from scratch:** start training an instance of the autoencoder model from scratch. 

- **train from checkpoint:** resume training from a pretrained checkpoint for an additional amount of epochs, using pretrained model settings and data.  

- **model evaluation:** evaluate the performance of pretrained model checkpoints using different metrics. The average mean squared error and mean average error are calculated for both the training and validation datasets. Random images are sampled from both datasets and reconstructed using a checkpoint encoder, while being visually compared to their original counterpart.   

**3) Encode images:** select a model checkpoint and use it to encode images into an abstract representation of the most relevant features. These low-dimension embeddings are saved as .npy files in *resources/inference*. 

**4) Setup and Maintenance:** execute optional commands such as *Install project into environment* to reinstall the project within your environment, *update project* to pull the last updates from github, and *remove logs* to remove all logs saved in *resources/logs*. 

**5) Exit:** close the program immediately 

### 5.2 Resources
This folder organizes data and results across various stages of the project, such as data validation, model training, and evaluation. The directory structure includes the following folders:

- **checkpoints:** pretrained model checkpoints are stored here, and can be loaded either for resuming training or use them for inference.

- **database:** Processed data and validation results will be stored centrally within the main database *FEXT_database.db*. All associated metadata will be promptly stored in *database/metadata*. For image training data, ensure all image files are placed in *database/images*, adhering to specified formats (.jpeg or .png). Graphical validation outputs will be saved separately within *database/validation*.

- **inference:** contains images intended as input for inference using a pretrained checkpoint. The resulting lower-dimension projections of these images are saved here as .npy files.

- **logs:** log files are saved here

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
| IMG_AUGMENTATION   | Whether to apply data augmentation to images             |
| SPLIT_SEED         | Seed for random splitting of the dataset                 |
| SAVE_CSV           | Save preprocessed data as .csv file                      |

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

#### Training Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| EPOCHS             | Number of epochs to train the model                      |
| ADDITIONAL EPOCHS  | Number of epochs to train the model from checkpoint      |
| BATCH_SIZE         | Number of samples per batch                              |
| USE_TENSORBOARD    | Whether to use TensorBoard for logging                   |
| SAVE_CHECKPOINTS   | Save checkpoints during training (at each epoch)         |

#### LR Scheduler Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| INITIAL_LR         | Initial value of learni rate                             |
| CONSTANT_STEPS     | Number of steps (batch) to keep the learning rate stable |
| DECAY_STEPS        | Number of steps (batch) to decay learning rate           |

#### Validation Configuration

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| BATCH_SIZE         | Number of samples per batch                              |
| NUM_IMAGES         | Max number of images to compare during evaluation        |
| DPI                | Resolution of figures from validation                    |

 
**Environmental variables** are stored in *setup/variables/.env*. For security reasons, this file is typically not uploaded to GitHub. Instead, you must create this file manually by copying the template from *resources/templates/.env* and placing it in the *setup/variables* directory.

| Variable              | Description                                              |
|-----------------------|----------------------------------------------------------|
| KERAS_BACKEND         | Sets the backend for Keras, default is PyTorch           |
| TF_CPP_MIN_LOG_LEVEL  | TensorFlow logging verbosity                             |


## 7. License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.
