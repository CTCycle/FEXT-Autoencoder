import os
import re
import shutil
import random
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt

from FEXT.commons.utils.data.loader import ImageDataLoader
from FEXT.commons.utils.learning.callbacks import InterruptTraining
from FEXT.commons.utils.data.serializer import ModelSerializer
from FEXT.commons.interface.workers import check_thread_status, update_progress_callback
from FEXT.commons.constants import CHECKPOINT_PATH, EVALUATION_PATH 
from FEXT.commons.logger import logger


# [LOAD MODEL]
################################################################################
class ModelEvaluationSummary:

    def __init__(self, database, configuration, remove_invalid=False):
        self.remove_invalid = remove_invalid             
        self.database = database       
        self.configuration = configuration

    #---------------------------------------------------------------------------
    def scan_checkpoint_folder(self):
        model_paths = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():                
                pretrained_model_path = os.path.join(entry.path, 'saved_model.keras')                
                if os.path.isfile(pretrained_model_path):
                    model_paths.append(entry.path)
                elif not os.path.isfile(pretrained_model_path) and self.remove_invalid:                    
                    shutil.rmtree(entry.path)

        return model_paths  

    #---------------------------------------------------------------------------
    def get_checkpoints_summary(self, **kwargs):
        serializer = ModelSerializer()      
        model_paths = self.scan_checkpoint_folder()
        model_parameters = []            
        for i, model_path in enumerate(model_paths):            
            model = serializer.load_checkpoint(model_path)
            configuration, history = serializer.load_training_configuration(model_path)
            model_name = os.path.basename(model_path)                   
            precision = 16 if configuration.get("use_mixed_precision", 'NA') else 32 
            chkp_config = {'Checkpoint name': model_name,                                                  
                           'Sample size': configuration.get("train_sample_size", 'NA'),
                           'Validation size': configuration.get("validation_size", 'NA'),
                           'Seed': configuration.get("train_seed", 'NA'),                           
                           'Precision (bits)': precision,                      
                           'Epochs': configuration.get("epochs", 'NA'),
                           'Additional Epochs': configuration.get("additional_epochs", 'NA'),
                           'Batch size': configuration.get("batch_size", 'NA'),           
                           'Split seed': configuration.get("split_seed", 'NA'),
                           'Image augmentation': configuration.get("img_augmentation", 'NA'),
                           'Image height': 128,
                           'Image width': 128,
                           'Image channels': 3,                          
                           'JIT Compile': configuration.get("jit_compile", 'NA'),                           
                           'Device': configuration.get("device", 'NA'),                                                      
                           'Number workers': configuration.get("num_workers", 'NA'),
                           'LR Scheduler': configuration.get("use_scheduler", 'NA'),                                                      
                           'Initial LR': configuration.get("initial_LR", 'NA'),
                           'Constant steps': configuration.get("constant_steps", 'NA'),
                           'Decay steps': configuration.get("decay_steps", 'NA')}

            model_parameters.append(chkp_config)

            # check for thread status and progress bar update   
            check_thread_status(kwargs.get('worker', None))         
            update_progress_callback(
                i, len(model_paths), kwargs.get('progress_callback', None)) 

        dataframe = pd.DataFrame(model_parameters)
        self.database.save_checkpoints_summary_table(dataframe)    
            
        return dataframe
    
    #--------------------------------------------------------------------------
    def get_evaluation_report(self, model, validation_dataset, **kwargs):
        callbacks_list = [InterruptTraining(kwargs.get('worker', None))]
        validation = model.evaluate(validation_dataset, verbose=1, callbacks=callbacks_list)    
        logger.info(
            f'RMSE loss {validation[0]:.3f} - Cosine similarity {validation[1]:.3f}')     
    
    
# [IMAGE RECONSTRUCTION]
###############################################################################
class ImageReconstruction:

    def __init__(self, configuration, model, checkpoint_path):       
        self.checkpoint_name = os.path.basename(checkpoint_path)        
        self.validation_path = os.path.join(EVALUATION_PATH, self.checkpoint_name)       
        os.makedirs(self.validation_path, exist_ok=True)  

        self.num_images = configuration.get('num_evaluation_images', 6)
        self.DPI = 400 
        self.file_type = 'jpeg'
        self.model = model  
        self.configuration = configuration

    #--------------------------------------------------------------------------
    def save_image(self, fig, name):
        name = re.sub(r'[^0-9A-Za-z_]', '_', name)
        out_path = os.path.join(self.validation_path, name)
        fig.savefig(out_path, bbox_inches='tight', dpi=self.DPI)         

    #-------------------------------------------------------------------------- 
    def get_images(self, data):
        loader = ImageDataLoader(self.configuration)
        images = [loader.load_image(path, as_array=True) for path in 
                  random.sample(data, self.num_images)]
        norm_images = [loader.image_normalization(img) for img in images]
                
        return norm_images

    #-------------------------------------------------------------------------- 
    def visualize_3D_latent_space(self, model, dataset, num_images=10):
        # Extract latent representations
        pass
    
    #-------------------------------------------------------------------------- 
    def visualize_reconstructed_images(self, validation_data, **kwargs):             
        val_images = self.get_images(validation_data)
        logger.info(f'Comparing {self.num_images} reconstructed images from validation dataset')
        fig, axs = plt.subplots(self.num_images, 2, figsize=(4, self.num_images * 2))      
        for i, img in enumerate(val_images): 
                                 
            expanded_img = np.expand_dims(img, axis=0)                 
            reconstructed_image = self.model.predict(
                expanded_img, verbose=0, batch_size=1)[0]                          
            
            real = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            pred = np.clip(reconstructed_image * 255.0, 0, 255).astype(np.uint8)        
            axs[i, 0].imshow(real)
            axs[i, 0].set_title('Original Picture' if i == 0 else "")
            axs[i, 0].axis('off')            
            axs[i, 1].imshow(pred)
            axs[i, 1].set_title('Reconstructed Picture' if i == 0 else "")
            axs[i, 1].axis('off')

            # check for thread status and progress bar update
            check_thread_status(kwargs.get('worker', None))
            update_progress_callback(
                i, len(val_images), kwargs.get('progress_callback', None))
        
        plt.tight_layout()
        self.save_image(fig, 'images_recostruction.jpeg')
        plt.close() 

        return fig    
                 
