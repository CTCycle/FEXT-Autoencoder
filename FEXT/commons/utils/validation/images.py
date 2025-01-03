import os
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

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
    def visualize_latent_space(self, model : keras.Model, dataset, path, method='PCA', n_components=2):

        methods = {'PCA': lambda latent: PCA(n_components=n_components).fit_transform(latent),
                    'TSNE': lambda latent: TSNE(n_components=n_components, random_state=42).fit_transform(latent),
                    'UMAP': lambda latent: umap.UMAP(n_components=n_components, random_state=42).fit_transform(latent)}

        if method not in methods:
            raise ValueError(f"Invalid method '{method}'. Choose from {list(methods.keys())}.")

        if n_components not in [2, 3]:
            raise ValueError("n_components must be 2 or 3 for visualization.")

        # Extract latent representations
        latent_representations = model.predict(dataset)
        latent_representations = latent_representations.reshape(len(latent_representations), -1)

        # Apply the selected transformation
        reduced_latent = methods[method](latent_representations)

        # Plot the results
        if n_components == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(reduced_latent[:, 0], reduced_latent[:, 1], s=5, cmap='viridis')
            plt.title(f"Latent Representation ({method.upper()})")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
        elif n_components == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(reduced_latent[:, 0], reduced_latent[:, 1], reduced_latent[:, 2], s=5, cmap='viridis')
            ax.set_title(f"Latent Representation ({method.upper()})")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
    
    #-------------------------------------------------------------------------- 
    def visualize_reconstructed_images(self, dataset : tf.data.Dataset, path):
        
        # perform visual validation for the train dataset (initialize a validation tf.dataset
        # with batch size of 10 images)                
        batch = dataset.take(1)
        for images, labels in batch:                  
            reconstructed_images = self.model.predict(images, verbose=0)
            num_pics = len(images)        
            fig, axs = plt.subplots(num_pics, 2, figsize=(4, num_pics * 2))
            for i, (real, pred) in enumerate(zip(images, reconstructed_images)):      
                real = np.clip(real, 0, 1)
                pred = np.clip(pred, 0, 1)
                axs[i, 0].imshow(real)
                if i == 0:
                    axs[i, 0].set_title('Original Picture')
                axs[i, 0].axis('off')

                axs[i, 1].imshow(pred)
                if i == 0:
                    axs[i, 1].set_title('Reconstructed Picture')
                axs[i, 1].axis('off')

            plt.tight_layout()          


