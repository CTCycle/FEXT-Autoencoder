import os
import shutil
import pandas as pd

from FEXT.commons.utils.dataloader.serializer import ModelSerializer
from FEXT.commons.constants import CONFIG, CHECKPOINT_PATH, VALIDATION_PATH
from FEXT.commons.logger import logger



# [LOAD MODEL]
################################################################################
class ModelEvaluationSummary:

    def __init__(self, remove_invalid=False):
        self.remove_invalid = remove_invalid
        self.serializer = ModelSerializer()
        self.summary_path = os.path.join(VALIDATION_PATH, 'checkpoints')
        os.makedirs(self.summary_path, exist_ok=True)

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
    def checkpoints_summary(self):       
        # look into checkpoint folder to get pretrained model names      
        model_paths = self.scan_checkpoint_folder()
        model_parameters = []            
        for model_path in model_paths:            
            model = self.serializer.load_checkpoint(model_path)
            configuration, history = self.serializer.load_session_configuration(model_path)
            model_name = os.path.basename(model_path) 
            # Extract model name and training type                       
            device_config = configuration["device"]
            precision = 16 if device_config.get("use_mixed_precision", 'NA') == True else 32
            chkp_config = {'Checkpoint name': model_name,                                                 
                           'Sample size': configuration["dataset"].get("SAMPLE_SIZE", 'NA'),
                           'Validation size': configuration["dataset"].get("VALIDATION_SIZE", 'NA'),
                           'Seed': configuration.get("SEED", 'NA'),                          
                           'Precision (bits)': precision,                     
                           'Epochs': configuration["training"].get("EPOCHS", 'NA'),
                           'Learning rate': configuration["training"].get("LEARNING_RATE", 'NA'),
                           'Batch size': configuration["training"].get("BATCH_SIZE", 'NA'),                          
                           'Normalize': configuration["dataset"].get("IMG_NORMALIZE", 'NA'),
                           'Split seed': configuration["dataset"].get("SPLIT_SEED", 'NA'),
                           'Image augment': configuration["dataset"].get("IMG_AUGMENT", 'NA'),                          
                           'Residuals': configuration["model"].get("RESIDUAL_CONNECTIONS", 'NA'),
                           'JIT Compile': configuration["model"].get("JIT_COMPILE", 'NA'),
                           'JIT Backend': configuration["model"].get("JIT_BACKEND", 'NA'),
                           'Device': configuration["device"].get("DEVICE", 'NA'),
                           'Device ID': configuration["device"].get("DEVICE_ID", 'NA'),
                           'Number of Processors': configuration["device"].get("NUM_PROCESSORS", 'NA'),
                           'Tensorboard logs': configuration["training"].get("USE_TENSORBOARD", 'NA')}

            model_parameters.append(chkp_config)

        # Define the CSV path
        dataframe = pd.DataFrame(model_parameters)
        csv_path = os.path.join(self.summary_path, 'checkpoints_summary.csv')     
        dataframe.to_csv(csv_path, index=False, sep=';', encoding='utf-8')        
            
        return dataframe
    
    
