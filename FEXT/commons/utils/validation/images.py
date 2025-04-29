import os
import cv2
import random
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from FEXT.commons.utils.data.loader import InferenceDataLoader
from FEXT.commons.utils.data.database import FEXTDatabase
from FEXT.commons.constants import DATA_PATH, VALIDATION_PATH
from FEXT.commons.logger import logger


# [IMAGE RECONSTRUCTION]
###############################################################################
class ImageReconstruction:

    def __init__(self, configuration : dict, model : keras.Model, checkpoint_path : str):       
        self.checkpoint_name = os.path.basename(checkpoint_path)        
        self.validation_path = os.path.join(VALIDATION_PATH, self.checkpoint_name)       
        os.makedirs(self.validation_path, exist_ok=True)
        self.loader = InferenceDataLoader(configuration) 

        self.model = model  
        self.configuration = configuration
        self.num_images = configuration['validation']['NUM_IMAGES']
        self.DPI = configuration['validation']['DPI']  
        self.file_type = 'jpeg'

    #-------------------------------------------------------------------------- 
    def get_images(self, data):
        images = [self.loader.load_image_as_array(path) for path in 
                  random.sample(data, self.num_images)]        
                
        return images

    #-------------------------------------------------------------------------- 
    def visualize_3D_latent_space(self, model : keras.Model, dataset : tf.data.Dataset, num_images=10):
        # Extract latent representations
        latent_representations = model.predict(dataset)
        latent_representations = latent_representations.reshape(len(latent_representations), -1)
        # Apply the selected transformation
        reduced_latent = PCA(n_components=3).fit_transform(latent_representations)       
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_latent[:, 0], reduced_latent[:, 1], reduced_latent[:, 2], s=5, cmap='viridis')
        ax.set_title(f"Latent Representation (PCA) of {num_images} images")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        plt.savefig(
            os.path.join(self.validation_path, 'PCA.jpeg'), 
            dpi=self.DPI)
    
    #-------------------------------------------------------------------------- 
    def visualize_reconstructed_images(self, validation_data):       
        val_images = self.get_images(validation_data)
        logger.info(
        f'Comparing {self.num_images} reconstructed images from validation dataset')
        fig, axs = plt.subplots(self.num_images, 2, figsize=(4, self.num_images * 2))      
        for i, img in enumerate(val_images):           
            expanded_img = np.expand_dims(img, axis=0)                 
            reconstructed_image = self.model.predict(
                expanded_img, verbose=0, batch_size=1)[0]              
            real = np.clip(img, 0, 1)
            pred = np.clip(reconstructed_image, 0, 1)          
            axs[i, 0].imshow(real)
            axs[i, 0].set_title('Original Picture' if i == 0 else "")
            axs[i, 0].axis('off')            
            axs[i, 1].imshow(pred)
            axs[i, 1].set_title('Reconstructed Picture' if i == 0 else "")
            axs[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.validation_path, 'images_recostruction.jpeg'), 
            dpi=self.DPI)
        plt.close()    
                 

# [VALIDATION OF PRETRAINED MODELS]
###############################################################################
class ImageAnalysis:

    def __init__(self, configuration):                          
        self.DPI = configuration['validation']['DPI']  
        self.configuration = configuration 
        self.database = FEXTDatabase(configuration)       
        
    #--------------------------------------------------------------------------
    def calculate_image_statistics(self, images_path : list):          
        results= []     
        for path in tqdm(
            images_path, desc="Processing images", total=len(images_path), ncols=100):                  
            img = cv2.imread(path)            
            if img is None:
                logger.warning(f"Warning: Unable to load image at {path}.")
                continue

            # Convert image to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Get image dimensions
            height, width = gray.shape
            # Compute basic statistics
            mean_val = np.mean(gray)
            median_val = np.median(gray)
            std_val = np.std(gray)
            min_val = np.min(gray)
            max_val = np.max(gray)
            pixel_range = max_val - min_val
            # Estimate noise by comparing the image to a blurred version
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            noise = gray.astype(np.float32) - blurred.astype(np.float32)
            noise_std = np.std(noise)
            # Define the noise ratio (avoiding division by zero with a small epsilon)
            noise_ratio = noise_std / (std_val + 1e-9)          
            results.append({'name': os.path.basename(path),
                            'height': height,
                            'width': width,
                            'mean': mean_val,
                            'median': median_val,
                            'std': std_val,
                            'min': min_val,
                            'max': max_val,
                            'pixel_range': pixel_range,
                            'noise_std': noise_std,
                            'noise_ratio': noise_ratio})    

        stats_dataframe = pd.DataFrame(results) 
        self.database.save_image_statistics_table(stats_dataframe)       
        
        return stats_dataframe
    
    #--------------------------------------------------------------------------
    def calculate_pixel_intensity_distribution(self, images_path : list, progress_callback=None):                
        image_histograms = np.zeros(256, dtype=np.int64)        
        for i, path in enumerate(
            tqdm(images_path, desc="Processing image histograms", 
            total=len(images_path), ncols=100)):
                        
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Warning: Unable to load image at {path}.")
                continue

            # Calculate histogram for grayscale values [0, 255]
            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            image_histograms += hist.astype(np.int64)

            if progress_callback is not None:
                total = len(images_path)
                percent = int((i + 1) * 100 / total)
                progress_callback(percent)

        # Plot the combined histogram
        plt.figure(figsize=(14, 12))
        plt.bar(np.arange(256),image_histograms, alpha=0.7)
        plt.title('Combined Pixel Intensity Histogram', fontsize=16)
        plt.xlabel('Pixel Intensity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(VALIDATION_PATH, 'pixel_intensity_histogram.jpeg'), 
            dpi=self.DPI)
        plt.close()        

        return image_histograms
    
    #--------------------------------------------------------------------------
    def compare_train_and_validation_PID(self, train_images_path: list, val_images_path: list):
        # Initialize histograms for training and validation images
        train_hist = np.zeros(256, dtype=np.int64)
        val_hist = np.zeros(256, dtype=np.int64)

        # Process training images
        for path in tqdm(train_images_path, desc="Processing training images", total=len(train_images_path), ncols=100):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Warning: Unable to load training image at {path}.")
                continue
            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            train_hist += hist.astype(np.int64)

        # Process validation images
        for path in tqdm(val_images_path, desc="Processing validation images", total=len(val_images_path), ncols=100):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Warning: Unable to load validation image at {path}.")
                continue
            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            val_hist += hist.astype(np.int64)

        # Plot the histograms overlapped
        plt.figure(figsize=(14, 12))
        # Offset the positions slightly so that the bars don't completely overlap
        x = np.arange(256)
        width = 0.4  # width of each bar
        plt.bar(x - width/2, train_hist, width=width, label='Train', alpha=0.7)
        plt.bar(x + width/2, val_hist, width=width, label='Validation', alpha=0.7)
        plt.title('Pixel Intensity Histogram Comparison', fontsize=16)
        plt.xlabel('Pixel Intensity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(VALIDATION_PATH, 'pixel_intensity_histogram_comparison.jpeg'),
                    dpi=self.DPI)
        plt.close()

        return train_hist, val_hist

