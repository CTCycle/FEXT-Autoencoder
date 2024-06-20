import os
from tensorflow import keras

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

    
# [CALLBACK FOR REAL TIME TRAINING MONITORING]
#------------------------------------------------------------------------------
class RealTimeHistory(keras.callbacks.Callback):
    
    '''
    Custom Keras callback to visualize training and validation metrics in real-time.

    This callback logs training and validation metrics (loss and metrics) after each epoch
    and periodically generates plots of these metrics. It saves the plots as JPEG images 
    in the specified directory.

    Parameters:    
        plot_path : str
            Directory path where the plots will be saved.
        update_frequency : int, optional (default=2)
            Frequency (in epochs) at which to update the logging of metrics.
        plot_frequency : int, optional (default=5)
            Frequency (in epochs) at which to generate and save plots.
        validation : bool, optional (default=True)
            Whether to log and plot validation metrics.

    Methods:    
        on_epoch_end(epoch, logs)
            Method called by Keras at the end of each epoch to update metrics and potentially generate plots.

        plot_training_history()
            Generates and saves plots of training and validation metrics.

    '''    
    def __init__(self, plot_path, update_frequency=2, 
                 plot_frequency=5, validation=True):
        super().__init__()
        self.plot_path = plot_path
        self.update_frequency = update_frequency
        self.plot_frequency = plot_frequency
        self.validation = validation 
        
        # Initialize dictionaries to store history
        self.history = {}
        self.val_history = {}
        
        # Ensure plot directory exists
        os.makedirs(self.plot_path, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs={}):
        # Log metrics and losses
        for key, value in logs.items():
            if key.startswith('val_'):
                if self.validation:
                    if key not in self.val_history:
                        self.val_history[key] = []
                    self.val_history[key].append(value)
            else:
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
        
        # Update plots if necessary
        if epoch % self.plot_frequency == 0:
            self.plot_training_history()

    def plot_training_history(self):
        fig_path = os.path.join(self.plot_path, 'training_history.jpeg')
        plt.figure(figsize=(10, 8))
        
        # Plot each metric
        for i, (metric, values) in enumerate(self.history.items()):
            plt.subplot(len(self.history), 1, i + 1)
            plt.plot(range(len(values)), values, label=f'train {metric}')
            if self.validation and f'val_{metric}' in self.val_history:
                plt.plot(range(len(self.val_history[f'val_{metric}'])), self.val_history[f'val_{metric}'], label=f'val {metric}')
                plt.legend(loc='best', fontsize=8)
            plt.title(f'{metric} Plot')
            plt.ylabel(metric)
            plt.xlabel('Epoch')
        
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches='tight', format='jpeg', dpi=300)
        plt.close()


# [LEARNING RATE SCHEDULER]
#------------------------------------------------------------------------------
@keras.utils.register_keras_serializable(package='LRScheduler')
class LRScheduler(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, decay_steps, decay_rate, warmup_steps=0):
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        self.warmup_lr = initial_lr * warmup_steps
        
    # call on step
    #--------------------------------------------------------------------------
    def __call__(self, step):               
        step = step + 1
        step_tensor = tf.convert_to_tensor(step, dtype=tf.float32)
        if self.warmup_steps > 0:
            warmup_lr = self.warmup_lr * (step_tensor/self.warmup_steps)
        else:
            warmup_lr = self.initial_lr

        decay_lr = self.initial_lr * (self.decay_rate ** ((step - self.warmup_steps) // self.decay_steps))
        lr = tf.cond(tf.math.less(step_tensor, self.warmup_steps),
                     lambda: warmup_lr,
                     lambda: decay_lr)
        
        return lr
    
    # custom configurations
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(LRScheduler, self).get_config()
        config.update({'initial_lr': self.initial_lr,
                       'decay_steps': self.decay_steps,
                       'decay_rate': self.decay_rate,
                       'warmup_steps': self.warmup_steps})
        return config        
    
    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    