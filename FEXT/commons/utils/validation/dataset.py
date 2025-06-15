import os
import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from FEXT.commons.utils.data.loader import InferenceDataLoader
from FEXT.commons.interface.workers import check_thread_status, update_progress_callback
from FEXT.commons.constants import EVALUATION_PATH
from FEXT.commons.logger import logger


# [IMAGE RECONSTRUCTION]
###############################################################################
class ImageReconstruction:

    def __init__(self, configuration, model, checkpoint_path):       
        self.checkpoint_name = os.path.basename(checkpoint_path)        
        self.EVALUATION_PATH = os.path.join(EVALUATION_PATH, self.checkpoint_name)       
        os.makedirs(self.EVALUATION_PATH, exist_ok=True)  

        self.num_images = configuration.get('num_evaluation_images', 6)
        self.DPI = 400 
        self.file_type = 'jpeg'
        self.model = model  
        self.configuration = configuration

    #-------------------------------------------------------------------------- 
    def get_images(self, data):
        loader = InferenceDataLoader(self.configuration)
        images = [loader.load_image_as_array(path) for path in 
                  random.sample(data, self.num_images)]        
                
        return images

    #-------------------------------------------------------------------------- 
    def visualize_3D_latent_space(self, model, dataset, num_images=10):
        # Extract latent representations
        latent_representations = model.predict(dataset)
        latent_representations = latent_representations.reshape(
            len(latent_representations), -1)
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
            os.path.join(self.EVALUATION_PATH, 'PCA.jpeg'), 
            dpi=self.DPI)
    
    #-------------------------------------------------------------------------- 
    def visualize_reconstructed_images(self, validation_data, **kwargs):        
        val_images = self.get_images(validation_data)
        logger.info(f'Comparing {self.num_images} reconstructed images from validation dataset')
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

            # check for thread status and progress bar update
            check_thread_status(kwargs.get('worker', None))
            update_progress_callback(
                i, val_images, kwargs.get('progress_callback', None))
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.EVALUATION_PATH, 'images_recostruction.jpeg'), 
            dpi=self.DPI)
        plt.close() 

        return fig    
                 

# [VALIDATION OF PRETRAINED MODELS]
###############################################################################
class ImageAnalysis:

    def __init__(self, database, configuration):           
        self.database = database
        self.save_images = configuration.get('save_images', True)          
        self.configuration = configuration      
        self.DPI = 400 

    #--------------------------------------------------------------------------
    def save_image(self, fig, name):        
        out_path = os.path.join(EVALUATION_PATH, name)
        fig.savefig(out_path, bbox_inches='tight', dpi=self.DPI)     
        
    #--------------------------------------------------------------------------
    def calculate_image_statistics(self, images_path, **kwargs):          
        results= []     
        for i, path in enumerate(tqdm(
            images_path, desc="Processing images", total=len(images_path), ncols=100)):                  
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

            # check for thread status and progress bar update
            check_thread_status(kwargs.get('worker', None))
            update_progress_callback(
                i, images_path, kwargs.get('progress_callback', None))  

        stats_dataframe = pd.DataFrame(results) 
        self.database.save_image_statistics_table(stats_dataframe)       
        
        return stats_dataframe
    
    #--------------------------------------------------------------------------
    def calculate_pixel_intensity_distribution(self, images_path, **kwargs):                 
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
            
            # check for thread status and progress bar update
            check_thread_status(kwargs.get('worker', None))
            update_progress_callback(
                i, images_path, kwargs.get('progress_callback', None))  

        # Plot the combined histogram
        fig, ax = plt.subplots(figsize=(16, 14), dpi=self.DPI)
        plt.bar(np.arange(256),image_histograms, alpha=0.7)
        ax.set_title('Combined Pixel Intensity Histogram', fontsize=20)
        ax.set_xlabel('Pixel Intensity', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        plt.tight_layout()        
        self.save_image(fig, "pixels_intensity_histogram.jpeg") if self.save_images else None
        plt.close()          

        return fig              
    
    #--------------------------------------------------------------------------
    def compare_train_and_validation_PID(self, train_images_path, val_images_path, **kwargs):                
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
        for path in tqdm(val_images_path, desc="Processing validation images", 
                         total=len(val_images_path), ncols=100):
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
        plt.savefig(
            os.path.join(EVALUATION_PATH, 'pixel_intensity_histogram_comparison.jpeg'),
            dpi=self.DPI)
        plt.close()

        return train_hist, val_hist

