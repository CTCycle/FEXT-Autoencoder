import os
import numpy as np
import matplotlib.pyplot as plt


# [VALIDATION OF DATA]
###############################################################################
class DataValidation:

    def __init__(self):
        self.DPI = 400
        self.file_type = 'jpeg'

    #--------------------------------------------------------------------------
    def pixel_intensity_histograms(self, image_dict, path):

        figure_path = os.path.join(path, 'pixel_intensity_histograms.jpeg')
        plt.figure(figsize=(16, 14))        
        for name, image_set in image_dict.items():
            pixel_intensities = np.concatenate([image.flatten() for image in image_set])
            plt.hist(pixel_intensities, bins='auto', alpha=0.5, label=name)        
        plt.title('Pixel Intensity Histograms', fontsize=16)
        plt.xlabel('Pixel Intensity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.tight_layout()        
        plt.savefig(figure_path, bbox_inches='tight', format=self.file_type, dpi=self.DPI)
        plt.show()
        plt.close()
        

# [VALIDATION OF PRETRAINED MODELS]
###############################################################################
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

        eval_path = os.path.join(path, 'data')
        num_pics = len(real_images)
        fig_path = os.path.join(eval_path, f'{name}.jpeg')
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
        

              
        
        
            
        
