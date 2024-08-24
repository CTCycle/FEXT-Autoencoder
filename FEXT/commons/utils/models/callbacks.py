import os
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from FEXT.commons.constants import CONFIG
from FEXT.commons.logger import logger

    
# [CALLBACK FOR REAL TIME TRAINING MONITORING]
###############################################################################
class RealTimeHistory(keras.callbacks.Callback):    
        
    def __init__(self, plot_path, past_logs=None, **kwargs):
        super(RealTimeHistory, self).__init__(**kwargs)
        self.plot_path = plot_path 
        self.past_logs = past_logs       
        self.plot_epoch_gap = CONFIG["training"]["PLOT_EPOCH_GAP"]
                
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
        
        # Update plots if necessary
        if epoch % self.plot_epoch_gap == 0:
            self.plot_training_history()

    #--------------------------------------------------------------------------
    def plot_training_history(self):
        fig_path = os.path.join(self.plot_path, 'training_history.jpeg')
        plt.figure(figsize=(10, 8))
        
        # Plot each metric
        for i, (metric, values) in enumerate(self.history.items()):
            plt.subplot(len(self.history), 1, i + 1)
            plt.plot(range(len(values)), values, label=f'train {metric}')
            if f'val_{metric}' in self.val_history:
                plt.plot(range(len(self.val_history[f'val_{metric}'])), self.val_history[f'val_{metric}'], label=f'val {metric}')
                plt.legend(loc='best', fontsize=8)
            plt.title(f'{metric} Plot')
            plt.ylabel(metric)
            plt.xlabel('Epoch')
        
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches='tight', format='jpeg', dpi=300)
        plt.close()



# [LOGGING]
###############################################################################
class LoggingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logger.debug(f"Epoch {epoch + 1}: {logs}")



    