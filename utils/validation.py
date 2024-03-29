import os
import numpy as np
import matplotlib.pyplot as plt


# [VALIDATION OF DATA]
#==============================================================================
# Series of methods and functions to preprocess data for model training
#==============================================================================
class DataValidation:

    def pixel_intensity_histograms(self, image_set_1, image_set_2, path, params,
                                   names=['First set', 'Second set']):
        
        pixel_intensities_1 = np.concatenate([image.flatten() for image in image_set_1])
        pixel_intensities_2 = np.concatenate([image.flatten() for image in image_set_2])        
        plt.hist(pixel_intensities_1, bins='auto', alpha=0.5, color='blue', label=names[0])
        plt.hist(pixel_intensities_2, bins='auto', alpha=0.5, color='red', label=names[1])
        plt.title(params['title'],)
        plt.xlabel('Pixel Intensity', fontsize=params['fontsize_labels'])
        plt.ylabel(params['ylabel'],  fontsize=params['fontsize_labels'])
        plt.legend()            
        plt.tight_layout()
        plot_loc = os.path.join(path, params['filename'])
        plt.savefig(plot_loc, bbox_inches='tight', format='jpeg', dpi=400)   
        plt.show()         
        plt.close()
        

# [VALIDATION OF PRETRAINED MODELS]
#==============================================================================
# Collection of methods for machine learning validation and model evaluation
#==============================================================================
class ModelValidation:

    def __init__(self, model):        
        self.model = model

    #-------------------------------------------------------------------------- 
    def visualize_features_vector(self, real_image, features, predicted_image, name, path):          

        
        fig_path = os.path.join(path, f'{name}.jpeg')
        fig, axs = plt.subplots(1, 3, figsize=(14, 20), dpi=600)                                     
        axs[0].imshow(real_image[0])
        axs[0].set_title('Original picture')
        axs[0].axis('off')
        axs[1].imshow(features)
        axs[1].set_title('Extracted features')
        axs[1].axis('off')
        axs[2].imshow(predicted_image[0])
        axs[2].set_title('Reconstructed picture')
        axs[2].axis('off')
        plt.tight_layout() 
        plt.show(block=False)       
        plt.savefig(fig_path, bbox_inches='tight', format='jpeg', dpi=400)               
        plt.close()
        
    
    #-------------------------------------------------------------------------- 
    def visualize_reconstructed_images(self, real_images, predicted_images, name, path):          

        num_pics = len(real_images)
        fig_path = os.path.join(path, f'{name}.jpeg')
        fig, axs = plt.subplots(num_pics, 2, figsize=(4, num_pics * 2))
        for i, (real, pred) in enumerate(zip(real_images, predicted_images)):                                                          
            axs[i, 0].imshow(real)
            if i == 0:
                axs[i, 0].set_title('Original picture')
            axs[i, 0].axis('off')
            axs[i, 1].imshow(pred)
            if i == 0:
                axs[i, 1].set_title('Reconstructed picture')
            axs[i, 1].axis('off')
        plt.tight_layout() 
        plt.show(block=False)       
        plt.savefig(fig_path, bbox_inches='tight', format='jpeg', dpi=400)               
        plt.close()
        

              
        
        
            
        
