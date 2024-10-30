import os
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

from FEXT.commons.utils.dataloader.serializer import DataSerializer
from FEXT.commons.constants import RESULTS_PATH
from FEXT.commons.logger import logger

# [VALIDATION OF PRETRAINED MODELS]
###############################################################################
class ImageReconstruction:

    def __init__(self, model : keras.Model):
        self.DPI = 400
        self.file_type = 'jpeg'        
        self.model = model    

    #-------------------------------------------------------------------------- 
    def visualize_features_vector(self, real_image, features, predicted_image, path):
        
        fig_path = os.path.join(path, 'visual_feature_vector.jpeg')
        fig, axs = plt.subplots(1, 3, figsize=(14, 20), dpi=600)                                     
        axs[0].imshow(real_image)
        axs[0].set_title('Original picture')
        axs[0].axis('off')
        axs[1].imshow(features)
        axs[1].set_title('Extracted features')
        axs[1].axis('off')
        axs[2].imshow(predicted_image)
        axs[2].set_title('Reconstructed picture')
        axs[2].axis('off')
        plt.tight_layout() 
        plt.show(block=False)       
        plt.savefig(fig_path, bbox_inches='tight', format=self.file_type, dpi=self.DPI)                
        plt.close()
        
    
    #-------------------------------------------------------------------------- 
    def visualize_reconstructed_images(self, dataset : tf.data.Dataset, name, path):

        # perform visual validation for the train dataset (initialize a validation tf.dataset
        # with batch size of 10 images)
        logger.info('Visual reconstruction evaluation: train dataset')        
        batch = dataset.take(1)
        for images, labels in batch:
            recostructed_images = self.model.predict(images, verbose=0)  
            eval_path = os.path.join(path, 'data')
            num_pics = len(images)
            fig_path = os.path.join(eval_path, f'{name}.jpeg')
            fig, axs = plt.subplots(num_pics, 2, figsize=(4, num_pics * 2))
            for i, (real, pred) in enumerate(zip(images, recostructed_images)):                                                          
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
            plt.savefig(fig_path, bbox_inches='tight', format=self.file_type, dpi=self.DPI)               
            plt.close()



# [VALIDATION OF DATA]
###############################################################################
class ImageDatasetValidation:

    def __init__(self, train_data, validation_data, configuration):
        self.DPI = 400
        self.file_type = 'jpeg'
        self.train_data = train_data
        self.validation_data = validation_data
        self.serializer = DataSerializer(configuration)        

    #--------------------------------------------------------------------------
    def get_images_for_validation(self):

        train_images = (self.serializer.load_image(pt) 
                        for pt in self.train_data)
        validation_images = (self.serializer.load_image(pt) 
                             for pt in self.validation_data)

        return {'train' : train_images, 'validation' : validation_images}

    #--------------------------------------------------------------------------
    def pixel_intensity_histogram(self):

        images = self.get_images_for_validation()
        figure_path = os.path.join(RESULTS_PATH, 'pixel_intensity_histogram.jpeg')
        plt.figure(figsize=(16, 14))        
        for name, image_set in images.items():
            pixel_intensities = np.concatenate([image.flatten() for image in image_set])
            plt.hist(pixel_intensities, bins='auto', alpha=0.5, label=name)        
        plt.title('Pixel Intensity Histograms', fontsize=16)
        plt.xlabel('Pixel Intensity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.tight_layout()        
        plt.savefig(figure_path, bbox_inches='tight', 
                    format=self.file_type, dpi=self.DPI)
        plt.show()
        plt.close()
        

