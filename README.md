# FeXT AutoEncoder: Extraction of Images Features

## 1. Project Overview
The FeXT AutoEncoder project is centered around the development, evaluation, and application of a Convolutional AutoEncoder (CAE) model specifically designed for efficient image feature extraction. The architecture of this model draws inspiration from the renowned VGG16 model, a deep learning framework widely utilized in various computer vision tasks such as image reconstruction, anomaly detection, and feature extraction. As such, the model proposed in this project implements a series of stacked convolution layers, where pooling operations are performed to decrease the encoding dimensions recursively. Despite being similar to VVG16, the encoder submodel can optionally integrate a Sobel filter layer, which computes the pixels gradient and join this information with a parallel 2D convolution stream. Both the encoder and the decoder collaboratively work to extract salient features from input images, compressing the information into a lower-dimensional representation suitable for a wide range of downstream tasks.

![VVG16 encoder](FEXT/commons/assets/VGG16_encoder.png)
Architecture of the VVG16 encoder

## 2. FeXT AutoEncoder model
The encoder component of the FeXT AutoEncoder is responsible for image encoding into a lower-dimension latent space. It achieves this through a series of convolutional layers with a kernel size of 3x3 and a stride of 1 pixel. The kernel size is chosen to be compatible with the implementation of the Sobel filter layer, optionally used to extract information about the pixel gradients and use them in conjunction with the default convolution flow, with the downstream convolution layers being followed by max pooling operations. This allows to progressively downsample the spatial dimensions of the input image while expanding the channel dimensions (depth), effectively capturing the abstract representations of the image content. Each stack of convolutional layers is parametrized to use residual connections with layer normalization.

In contrast, the decoder is responsible for reconstructing the original image from the lower-dimensional encoded representation. This is achieved using transposed 2D convolutions and direct upsampling with 3x3 kernels. The decoder aims to restore the spatial dimensions and pixel details of the original image as faithfully as possible by leveraging the abstract features encoded by the model. A modified Mean Squared Error (MSE) function, incorporating a size-based penalty, is used to prevent the model from converging to suboptimal solutions, such as reconstructing an average-like image that lacks meaningful features (i.e., poor reconstruction)


## 3. Training dataset
The FeXT AutoEncoder model has been trained and tested on the Flickr 30K dataset (https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset), a comprehensive collection of images commonly used in many computer vision tasks. The versatility of the FeXT AutoEncoder allows it to be trained on any appropriately preprocessed image dataset, making it adaptable to a wide range of image data and tasks.

## 4. Installation
The installation process on Windows has been designed for simplicity and ease of use. To begin, simply run *start_on_windows.bat.* On its first execution, the installation procedure will automatically start with minimal user input required. The script will check if either Anaconda or Miniconda is installed on your system. If neither is found, it will automatically download and install the latest Miniconda release from https://docs.anaconda.com/miniconda/. After setting up Anaconda/Miniconda, the installation script will proceed with the installation of all necessary Python dependencies. This includes Keras 3 (with PyTorch support as the backend) and the required CUDA dependencies (CUDA 12.1) to enable GPU acceleration. If you'd prefer to handle the installation process separately, you can run the standalone installer by executing *setup/install_on_windows.bat*.

**Important:** After installation, if the project folder is moved or its path is changed, the application will no longer function correctly. To fix this, you can either:

- Open the main menu, select *Setup and maintentance* and choose *Install project in editable mode*
- Manually run the following commands in the terminal, ensuring the project folder is set as the current working directory (CWD):

    `conda activate FEXT`

    `pip install -e . --use-pep517` 

### 4.1 Just-In-Time (JIT) Compiler
This project leverages Just-In-Time model compilation through `torch.compile`, enhancing model performance by tracing the computation graph and applying advanced optimizations like kernel fusion and graph lowering. This approach significantly reduces computation time during both training and inference. The default backend, TorchInductor, is designed to maximize performance on both CPUs and GPUs. Additionally, the installation includes Triton, which generates highly optimized GPU kernels for even faster computation on NVIDIA hardware. For Windows users, a precompiled Triton wheel is bundled with the installation, ensuring seamless integration and performance improvements.

## 5. How to use
On Windows, run *start_on_windows.bat* to launch the main navigation menu and browse through the various options. Alternatively, each file can be executed individually by running *python path/filename.py* for Python scripts or *jupyter notebook path/notebook.ipynb* for Jupyter notebooks. Please note that some antivirus software, such as Avast, may flag or quarantine python.exe when called by the .bat file. If you encounter unusual behavior, consider adding an exception for your Anaconda or Miniconda environments in your antivirus settings.

### 5.1 Navigation menu

**1) Analyze image dataset:** runs *validation/image_dataset_validation.ipynb* to perform data validation using a series of metrics for image statistics. 

**2) Model training and evaluation:** open the machine learning menu to explore various options for model training and validation. Once the menu is open, you will see different options:
- **train from scratch:** runs *training/model_training.py* to start training an instance of the autoencoder model from scratch. 
- **train from checkpoint:** runs *training/train_from_checkpoint.py* to start training a pretrained checkpoint for an additional amount of epochs, using pretrained model settings and data.  
- **model evaluation:** runs *validation/model_evaluation.ipynb* to evaluate the performance of pretrained model checkpoints using different metrics. 

**3) Encode images:** runs *inference/images_encoding.py* to select a model checkpoint and use it to extract abstract representation of image features in the form of lower-dimension embeddings, which will be saved as npy files. 

**4) Setup and Maintenance:** execute optional commands such as *Install project into environment* to run the developer model project installation, and **remove logs** to remove all logs saved in *resources/logs*. 

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
| IMG_SHAPE          | Shape of the input images (height, width, channels)      |
| APPLY_SOBEL        | Apply Sobel filter in the encoder model                  |
| RESIDUALS          | Apply residual connections in convolution layers         |
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
| LEARNING_RATE      | Learning rate for the optimizer                          |
| BATCH_SIZE         | Number of samples per batch                              |
| USE_TENSORBOARD    | Whether to use TensorBoard for logging                   |
| SAVE_CHECKPOINTS   | Save checkpoints during training (at each epoch)         |
            
 
## 7. License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.
