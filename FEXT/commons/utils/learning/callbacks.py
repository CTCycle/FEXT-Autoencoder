import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import keras
import webbrowser
import subprocess
import time

from FEXT.commons.logger import logger

# [CALLBACK FOR UI PROGRESS BAR]
###############################################################################
class ProgressBarCallback(keras.callbacks.Callback):
    def __init__(self, progress_callback, total_epochs):
        super().__init__()
        self.progress_callback = progress_callback
        self.total_epochs = total_epochs

    #--------------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs=None):
        percent = int(100 * (epoch + 1) / self.total_epochs)
        if self.progress_callback is not None:
            self.progress_callback(percent)

    
# [CALLBACK FOR REAL TIME TRAINING MONITORING]
###############################################################################
class RealTimeHistory(keras.callbacks.Callback):    
        
    def __init__(self, plot_path, past_logs=None, **kwargs):
        super(RealTimeHistory, self).__init__(**kwargs)
        self.plot_path = plot_path 
        self.past_logs = past_logs       
                        
        # Initialize dictionaries to store history 
        self.history = {}
        self.val_history = {}
        if past_logs is not None:
            self.history = past_logs['history']
            self.val_history = past_logs['val_history']      
        
        # Ensure plot directory exists
        os.makedirs(self.plot_path, exist_ok=True)
    
    #--------------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs={}):
        # Log metrics and losses
        for key, value in logs.items():
            if key.startswith('val_'):
                if key not in self.val_history:
                    self.val_history[key] = []
                self.val_history[key].append(value)
            else:
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)        
        
        self.plot_training_history()

    #--------------------------------------------------------------------------
    def plot_training_history(self):
        fig_path = os.path.join(self.plot_path, 'training_history.jpeg')
        plt.figure(figsize=(16, 14))      
        for i, (metric, values) in enumerate(self.history.items()):
            plt.subplot(len(self.history), 1, i + 1)
            plt.plot(range(len(values)), values, label=f'train')
            if f'val_{metric}' in self.val_history:
                plt.plot(range(len(self.val_history[f'val_{metric}'])), 
                         self.val_history[f'val_{metric}'], label=f'validation')
                plt.legend(loc='best', fontsize=8)
            plt.title(metric)
            plt.ylabel('')
            plt.xlabel('Epoch')
        
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches='tight', format='jpeg', dpi=300)
        plt.close()

  
# [CALLBACKS HANDLER]
###############################################################################
def initialize_callbacks_handler(configuration, checkpoint_path, history, progress_callback=None):
    total_epochs = configuration.get('epochs', 10)
    callbacks_list = [ProgressBarCallback(progress_callback, total_epochs)]
    if configuration.get('plot_training_metrics', False):
        callbacks_list.append(RealTimeHistory(checkpoint_path, past_logs=history))

    if configuration.get('use_tensorboard', False):
        logger.debug('Using tensorboard during training')
        log_path = os.path.join(checkpoint_path, 'tensorboard')
        callbacks_list.append(keras.callbacks.TensorBoard(
            log_dir=log_path, histogram_freq=1, write_images=True))          
        start_tensorboard_subprocess(log_path)      

    # Add a checkpoint saving callback
    if configuration.get('save_checkpoints', False):
        logger.debug('Adding checkpoint saving callback')
        checkpoint_filepath = os.path.join(checkpoint_path, 'model_checkpoint.weights.h5')
        callbacks_list.append(keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                              save_weights_only=True,  
                                                              monitor='val_loss',       
                                                              save_best_only=True,      
                                                              mode='auto',              
                                                              verbose=0))
    return callbacks_list


###############################################################################
def start_tensorboard_subprocess(log_dir):    
    tensorboard_command = ["tensorboard", "--logdir", log_dir, "--port", "6006"]
    subprocess.Popen(
        tensorboard_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)      
    time.sleep(5)            
    webbrowser.open("http://localhost:6006")       
        
    