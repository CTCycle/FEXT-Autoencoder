

###############################################################################
class Configurations:
    
    def __init__(self):
        self.configurations = {
            
            'general_seed': 42,
            'split_seed': 76,
            'training_seed': 42,
            'view_type' : 'plot',

            # Dataset
            'sample_size': 1.0,
            'validation_size': 0.2,
            'use_img_augmentation': True,

            # Model 
            'jit_compile': False,
            'jit_backend': 'inductor',

            # Device
            'device': 'GPU',
            'device_id': 0,
            'use_mixed_precision': False,
            'num_processors': 6,

            # Training
            'epochs': 3,
            'additional_epochs': 10,
            'batch_size': 20,
            'use_tensorboard': True,
            'plot_training_metrics' : True,
            'save_checkpoints': False,

            # Learning rate scheduler
            'use_scheduler' : True,
            'initial_lr': 0.001,
            'constant_steps': 40000,
            'decay_steps': 1000,
            'final_lr': 0.0001,

            # Validation
            'val_batch_size': 20,
            'num_images': 6,
            'dpi': 400,
        }

    #--------------------------------------------------------------------------  
    def get_configurations(self):
        return self.configurations
    
    #--------------------------------------------------------------------------
    def update_value(self, key: str, value: bool):       
        self.configurations[key] = value