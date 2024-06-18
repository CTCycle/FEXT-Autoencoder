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
        
        # Initialize lists to store history
        self.epochs = []
        self.loss_hist = []
        self.metric_hist = []
        self.loss_val_hist = []
        self.metric_val_hist = []
        
        # Ensure plot directory exists
        os.makedirs(self.plot_path, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs={}):

        # Log metrics and losses
        self.epochs.append(epoch)
        self.loss_hist.append(logs.get('loss'))
        self.metric_hist.append(logs.get('cosine_similarity'))
        
        if self.validation:
            self.loss_val_hist.append(logs.get('val_loss'))
            self.metric_val_hist.append(logs.get('val_cosine_similarity'))
        
        # Update plots if necessary
        if epoch % self.plot_frequency == 0:
            self.plot_training_history()

    def plot_training_history(self):
        fig_path = os.path.join(self.plot_path, f'training_history_epoch_{self.epochs[-1]}.jpeg')
        plt.figure(figsize=(10, 8))

        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(self.epochs, self.loss_hist, label='training loss')
        if self.validation:
            plt.plot(self.epochs, self.loss_val_hist, label='validation loss')
            plt.legend(loc='best', fontsize=8)
        plt.title('Loss Plot')
        plt.ylabel('Mean Square Error')
        plt.xlabel('Epoch')

        # Plot metrics
        plt.subplot(2, 1, 2)
        plt.plot(self.epochs, self.metric_hist, label='train metrics')
        if self.validation:
            plt.plot(self.epochs, self.metric_val_hist, label='validation metrics')
            plt.legend(loc='best', fontsize=8)
        plt.title('Metrics Plot')
        plt.ylabel('Cosine Similarity')
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
    