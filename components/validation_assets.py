import os
import numpy as np
import matplotlib.pyplot as plt


  

# [VALIDATION OF DATA]
#==============================================================================
# Series of methods and functions to preprocess data for model training
#==============================================================================
class DataValidation:

    def pixel_intensity_histograms(self, image_set_1, image_set_2, path,
                                   names=['First set', 'Second set']):
        
        pixel_intensities_1 = np.concatenate([image.flatten() for image in image_set_1])
        pixel_intensities_2 = np.concatenate([image.flatten() for image in image_set_2])        
        plt.hist(pixel_intensities_1, bins='auto', alpha=0.5, color='blue', label=names[0])
        plt.hist(pixel_intensities_2, bins='auto', alpha=0.5, color='red', label=names[1])
        plt.title('Pixel Intensity Histograms')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()            
        plt.tight_layout()
        plot_loc = os.path.join(path, 'pixel_intensities.jpeg')
        plt.savefig(plot_loc, bbox_inches='tight', format='jpeg', dpi=400)            
        plt.close()
        
              
        
        
        
            
        
