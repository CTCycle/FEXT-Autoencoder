import tensorflow as tf

from FEXT.commons.utils.dataloader.generators import DatasetGenerator
from FEXT.commons.constants import CONFIG
from FEXT.commons.logger import logger
    


# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
class TrainingDatasetBuilder:

    def __init__(self, configuration, shuffle=True, evaluate=False):
        self.generator = DatasetGenerator(configuration) 
        self.configuration = configuration
        self.shuffle = shuffle  
        self.mode = 'training' if evaluate else 'validation'          

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def compose_tensor_dataset(self, images, batch_size, buffer_size=tf.data.AUTOTUNE):
        num_samples = len(images) 
        batch_size = self.configuration[self.mode]["BATCH_SIZE"] if batch_size is None else batch_size
        dataset = tf.data.Dataset.from_tensor_slices(images)                
        dataset = dataset.map(self.generator.load_image, num_parallel_calls=buffer_size)        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        dataset = dataset.shuffle(buffer_size=num_samples) if self.shuffle else dataset 

        return dataset         
      
    #--------------------------------------------------------------------------
    def build_model_dataloader(self, train_data, validation_data, batch_size=None):       
        train_dataset = self.compose_tensor_dataset(train_data, batch_size)
        validation_dataset = self.compose_tensor_dataset(validation_data, batch_size)       

        return train_dataset, validation_dataset
    






   


    