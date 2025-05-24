import os
import shutil
import pandas as pd

from FEXT.commons.utils.learning.callbacks import InterruptTraining
from FEXT.commons.utils.data.database import FEXTDatabase
from FEXT.commons.utils.data.serializer import ModelSerializer
from FEXT.commons.interface.workers import check_thread_status, update_progress_callback
from FEXT.commons.constants import DATA_PATH, CHECKPOINT_PATH
from FEXT.commons.logger import logger


# [LOAD MODEL]
################################################################################
class ModelEvaluationSummary:

    def __init__(self, configuration, remove_invalid=False):
        self.remove_invalid = remove_invalid
        self.serializer = ModelSerializer()        
        self.database = FEXTDatabase(configuration)        
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
    def get_checkpoints_summary(self, progress_callback=None, worker=None):            
        # look into checkpoint folder to get pretrained model names      
        model_paths = self.scan_checkpoint_folder()
        model_parameters = []            
        for i, model_path in enumerate(model_paths):            
            model = self.serializer.load_checkpoint(model_path)
            configuration, history = self.serializer.load_training_configuration(model_path)
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
            check_thread_status(worker)         
            update_progress_callback(i, model_paths, progress_callback) 

        dataframe = pd.DataFrame(model_parameters)
        self.database.save_checkpoints_summary_table(dataframe)    
            
        return dataframe
    
    #--------------------------------------------------------------------------
    def evaluation_report(self, model, validation_dataset, progress_callback=None, worker=None):
        callbacks_list = [InterruptTraining(worker)]
        validation = model.evaluate(validation_dataset, verbose=1, callbacks=callbacks_list)    
        logger.info(
            f'RMSE loss {validation[0]:.3f} - Cosine similarity {validation[1]:.3f}')     
    
    
