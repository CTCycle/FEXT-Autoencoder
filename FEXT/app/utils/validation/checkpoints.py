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

from FEXT.app.utils.data.loader import ImageDataLoader
from FEXT.app.utils.learning.callbacks import LearningInterruptCallback
from FEXT.app.utils.data.serializer import DataSerializer, ModelSerializer
from FEXT.app.interface.workers import check_thread_status, update_progress_callback
from FEXT.app.constants import CHECKPOINT_PATH, EVALUATION_PATH 
from FEXT.app.logger import logger


# [LOAD MODEL]
################################################################################
class ModelEvaluationSummary:

    def __init__(self, configuration : dict):         
        self.configuration = configuration

    #---------------------------------------------------------------------------
    def scan_checkpoint_folder(self):
        model_paths = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():                
                pretrained_model_path = os.path.join(entry.path, 'saved_model.keras')                
                if os.path.isfile(pretrained_model_path):
                    model_paths.append(entry.path)                

        return model_paths  

    #---------------------------------------------------------------------------
    def get_checkpoints_summary(self, **kwargs):
        modser = ModelSerializer() 
        serializer = DataSerializer(self.configuration)             
        model_paths = self.scan_checkpoint_folder()
        model_parameters = []            
        for i, model_path in enumerate(model_paths):            
            model = modser.load_checkpoint(model_path)
            configuration, history = modser.load_training_configuration(model_path)
            model_name = os.path.basename(model_path)                   
            precision = 16 if configuration.get("use_mixed_precision", np.nan) else 32 
            has_scheduler = configuration.get('use_scheduler', False)
            scores = history.get('history', {})
            chkp_config = {
                    'checkpoint': model_name,
                    'sample_size': configuration.get('sample_size', np.nan),
                    'validation_size': configuration.get('validation_size', np.nan),
                    'seed': configuration.get('train_seed', np.nan),
                    'precision': precision,
                    'epochs': history.get('epochs', np.nan),                    
                    'batch_size': configuration.get('batch_size', np.nan),
                    'split_seed': configuration.get('split_seed', np.nan),
                    'image_augmentation': configuration.get('img_augmentation', np.nan),
                    'image_height': 128,  
                    'image_width': 128,
                    'image_channels': 3,
                    'jit_compile': configuration.get('jit_compile', np.nan),
                    'has_tensorboard_logs': configuration.get('use_tensorboard', np.nan),
                    'initial_LR': configuration.get('initial_LR', np.nan),
                    'constant_steps_LR': configuration.get('constant_steps', np.nan) if has_scheduler else np.nan,
                    'decay_steps_LR': configuration.get('decay_steps', np.nan) if has_scheduler else np.nan,
                    'target_LR': configuration.get('target_LR', np.nan) if has_scheduler else np.nan,                    
                    'initial_neurons': configuration.get('initial_neurons', np.nan),
                    'dropout_rate': configuration.get('dropout_rate', np.nan),
                    'train_loss': scores.get('loss', [np.nan])[-1], 
                    'val_loss': scores.get('val_loss', [np.nan])[-1],
                    'train_cosine_similarity': scores.get('cosine_similarity', [np.nan])[-1], 
                    'val_cosine_similarity': scores.get('val_cosine_similarity', [np.nan])[-1]}
            
            model_parameters.append(chkp_config)

            # check for thread status and progress bar update   
            check_thread_status(kwargs.get('worker', None))         
            update_progress_callback(
                i+1, len(model_paths), kwargs.get('progress_callback', None)) 

        dataframe = pd.DataFrame(model_parameters)
        serializer.save_checkpoints_summary(dataframe)    
            
        return dataframe
    
    #--------------------------------------------------------------------------
    def get_evaluation_report(self, model, validation_dataset, **kwargs):
        callbacks_list = [LearningInterruptCallback(kwargs.get('worker', None))]
        validation = model.evaluate(validation_dataset, verbose=1, callbacks=callbacks_list) 
        logger.info(f'Evaluation of pretrained model has been completed')   
        logger.info(f'RMSE loss {validation[0]:.3f}')
        logger.info(f'Cosine similarity {validation[1]:.3f}')     
    
    
# [IMAGE RECONSTRUCTION]
###############################################################################
class ImageReconstruction:

    def __init__(self, configuration, model, checkpoint_path): 
        self.num_images = configuration.get('num_evaluation_images', 6)
        self.DPI = configuration.get('image_resolution', 400)
        self.file_type = 'jpeg'
        self.model = model  
        self.configuration = configuration

        self.checkpoint = os.path.basename(checkpoint_path)        
        self.validation_path = os.path.join(EVALUATION_PATH, self.checkpoint)       
        os.makedirs(self.validation_path, exist_ok=True) 

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
                i+1, len(val_images), kwargs.get('progress_callback', None))
        
        plt.tight_layout()
        self.save_image(fig, 'images_recostruction.jpeg')
        plt.close() 

        return fig    
                 
