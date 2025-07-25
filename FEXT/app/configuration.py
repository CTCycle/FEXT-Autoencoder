import json
from FEXT.app.constants import DATA_PATH


###############################################################################
class Configuration:
    
    def __init__(self):
        self.configuration = { 
            # Dataset
            'seed': 42,
            'sample_size': 1.0,
            'validation_size': 0.2,
            'img_augmentation': False,
            'shuffle_dataset': True,
            'shuffle_size': 256,

            # Model 
            'initial_neurons': 64,
            'dropout_rate': 0.2,
            'jit_compile': False,
            'jit_backend': 'inductor',

            # Device
            'use_device_GPU': False,
            'device_id': 0,
            'use_mixed_precision': False,
            'num_workers': 0,

            # Training
            'split_seed': 42,
            'train_seed': 42, 
            'train_sample_size': 1.0,
            'epochs': 100,
            'additional_epochs': 10,
            'batch_size': 32,
            'plot_training_metrics' : True,
            'use_tensorboard': False,            
            'save_checkpoints': False,

            # Learning rate scheduler
            'use_scheduler' : False,
            'initial_LR': 0.001,
            'constant_steps': 1000,
            'decay_steps': 500,
            'target_LR': 0.0001,

            # Inference   
            'inference_batch_size': 32,
            'num_evaluation_images': 6,    

            # viewer
            'image_resolution' : 400,              
        }

    #--------------------------------------------------------------------------  
    def get_configuration(self):
        return self.configuration
    
    #--------------------------------------------------------------------------
    def update_value(self, key: str, value: bool):       
        self.configuration[key] = value

    #--------------------------------------------------------------------------
    def save_configuration_to_json(self, filepath: str):        
        with open(filepath, 'w') as f:
            json.dump(self.configuration, f, indent=4)