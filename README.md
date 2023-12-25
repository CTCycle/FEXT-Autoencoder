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

**Pretrain FEXT-AutoEncoder model**: this module run the autoencoder training using a subset of images found in the **images** folder. Thr training is performed based on given parameters in configurations.py, and once it is finished, the model weights are saved in the designated folder, together with additional informations about the training/model parameters, validation of training and test set and tensorboard analysis (if selected)

**Evaluate FEXT-AutoEncoder model**: this module evaluates any selected model (from those in the **models** folder) by analysing perfromance on train and test sets and overall accuracy of the trained DNN. 

**Extract features from images**: with this function, a pretrained model is selected to perform images extraction and save the compressed features vectors in a csv file

**Exit and close**: quit the program
                 
### Requirements
This application has been developed and tested using the following dependencies (Python 3.10.12):

- `keras==2.10.0`
- `matplotlib==3.7.2`
- `numpy==1.25.2`
- `pandas==2.0.3`
- `scikit-learn==1.3.0`
- `scipy==1.11.2`
- `seaborn==0.12.2`
- `tensorflow==2.10.0`
- `tqdm==4.66.1`
- `xlrd==2.0.1`
- `XlsxWriter==3.1.3`

These dependencies are specified in the provided `requirements.txt` file to ensure full compatibility with the application. 


