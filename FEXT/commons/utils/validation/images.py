import os
import cv2
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.restoration import estimate_sigma
from tqdm import tqdm

from FEXT.commons.constants import CONFIG, RESULTS_PATH
from FEXT.commons.logger import logger



# [VALIDATION OF PRETRAINED MODELS]
###############################################################################
class ImageReconstruction:

    def __init__(self, model : keras.Model):
        self.DPI = 400
        self.file_type = 'jpeg'        
        self.model = model    

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
    
    #-------------------------------------------------------------------------- 
    def visualize_reconstructed_images(self, images : list):        
        num_pics = len(images)
        fig, axs = plt.subplots(num_pics, 2, figsize=(4, num_pics * 2))
        for i, img in enumerate(images):           
            expanded_img = np.expand_dims(img, axis=0)                 
            reconstructed_image = self.model.predict(expanded_img, verbose=0, batch_size=1)[0]              
            real = np.clip(img, 0, 1)
            pred = np.clip(reconstructed_image, 0, 1)          
            axs[i, 0].imshow(real)
            axs[i, 0].set_title('Original Picture' if i == 0 else "")
            axs[i, 0].axis('off')            
            axs[i, 1].imshow(pred)
            axs[i, 1].set_title('Reconstructed Picture' if i == 0 else "")
            axs[i, 1].axis('off')
        
        plt.tight_layout()
                 

# [VALIDATION OF PRETRAINED MODELS]
###############################################################################
class ImageAnalysis:

    def __init__(self, images_path : list):
        self.images_path = images_path
        self.validation_path = os.path.join(RESULTS_PATH, 'images_statistics.csv')

    #--------------------------------------------------------------------------
    def calculate_pixel_intensity(self):        
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in self.images_path]
        pixel_intensities = np.concatenate([image.flatten() for image in tqdm(images)], dtype=np.float16)
        plt.figure(figsize=(14, 12)) 
        plt.hist(pixel_intensities, bins='auto', alpha=0.7, color='blue', label='Dataset')
        plt.title('Pixel Intensity Histogram', fontsize=16)
        plt.xlabel('Pixel Intensity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.tight_layout()  
        
    #--------------------------------------------------------------------------
    def calculate_image_statistics(self):        
        results = []
        for path in tqdm(
            self.images_path, desc="Processing Images", total=len(self.images_path), ncols=100):                  
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
        stats_dataframe.to_csv(self.validation_path, index=False, sep=';', encoding='utf-8')
        
        return results
    

