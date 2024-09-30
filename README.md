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
The installation process on Windows has been designed for simplicity and ease of use. To begin, simply run `FEXT_AutoEncoder.bat`. On its first execution, the installation procedure will automatically start with minimal user input required. The script will check if either Anaconda or Miniconda is installed on your system. If neither is found, you will need to install it manually. You can download and install Miniconda by following the instructions here: (https://docs.anaconda.com/miniconda/).

After setting up Anaconda/Miniconda, the installation script will install all the necessary Python dependencies. This includes Keras 3 (with PyTorch support as the backend) and the required CUDA dependencies (CUDA 12.1) to enable GPU acceleration. If you'd prefer to handle the installation process separately, you can run the standalone installer by executing `setup/FEXT_installer.bat`. You can also use a custom python environment by modifying `settings/launcher_configurations.ini` and setting use_custom_environment as true, while specifying the name of your custom environment.

**Important:** After installation, if the project folder is moved or its path is changed, the application will no longer function correctly. To fix this, you can either:

- Open the main menu, select "FEXT setup," and choose "Install project packages"
- Manually run the following commands in the terminal, ensuring the project folder is set as the current working directory (CWD):

    `conda activate FEXT`

    `pip install -e . --use-pep517` 

### 3.1 Additional Package for XLA Acceleration
XLA is designed to optimize computations for speed and efficiency, particularly beneficial when working with TensorFlow and other machine learning frameworks that support XLA. Since this project uses Keras 3 with PyTorch as backend, the approach for optimizing computations for speed and efficiency has shifted from XLA to PyTorch's native acceleration tools, particularly TorchScript (currently not implemented). 

For those who wish to use Tensorflow as backend, XLA acceleration can be globally enabled setting the `XLA_FLAGS` environmental variabile with the following value: `--xla_gpu_cuda_data_dir=path\to\XLA`, where `path\to\XLA` is the actual directory path to the folder containing the nvvm subdirectory (where the file `libdevice.10.bc` resides).

## 4. How to use
On Windows, run `FEXT_AutoEncoder.bat` to launch the main navigation menu and browse through the various options. Alternatively, you can run each file separately using `python path/filename.py` or `jupyter path/notebook.ipynb`. 

### 4.1 Navigation menu

**1) Data analysis:** perform data validation using a series of metrics for image statistics, running `validation/data_validation.ipynb`

**2) Model training and evaluation:** open the machine learning menu to explore various options for model training and validation. Once the menu is open, you will see different options:
- **train from scratch:** runs `training/model_training.py` to start training an instance of the FEXT model from scratch using the available data and parameters. 
- **train from checkpoint:** runs `training/train_from_checkpoint.py` to start training a pretrained FEXt checkpoint for an additional amount of epochs, using pretrained model settings and data.  
- **model evaluation:** evaluate the performance of pretrained model checkpoints using different metrics, thoruhg running the jupyter notebook `validation/model_validation.ipynb`.

**3) Extract features from images:** runs `inference/images_encoding.py` to select a model checkpoint and use it to extract abstract representation of image features in the form of lower-dimension embeddings, which will be saved as npy files. 

**4) Exit and close:** exit the program immediately

### 4.2 Resources
This folder is used to organize data and results for various stages of the project, including data validation, model training, and evaluation. Here are the key subfolders:

- **checkpoints:**  pretrained model checkpoints are stored here, and can be used either for resuming training or performing inference with an already trained model.

- **dataset:** This folder contains images used to train the autoencoder model. Ensure your training data is placed here, and that the images format is of valid type (preferably either .jpg or .png).

- **extraction:**
Contains `input images` where you place images intended as an input for inference using the pretrained encoder. Moreover, hosts the folder `image features` where the resulting lower-dimension embeddings of the input images are saved (as npy files).

- **logs:** the application logs are saved within this folder

- **validation:** Used to save the results of data validation processes. This helps in keeping track of validation metrics and logs.


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
