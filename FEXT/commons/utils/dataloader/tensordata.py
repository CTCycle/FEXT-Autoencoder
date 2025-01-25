import pandas as pd
import tensorflow as tf

from FEXT.commons.utils.dataloader.generators import DatasetGenerator
from FEXT.commons.constants import CONFIG
from FEXT.commons.logger import logger
    


# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
class TensorDatasetBuilder:

    def __init__(self, configuration):
        self.generator = DatasetGenerator(configuration) 
        self.configuration = configuration

    #--------------------------------------------------------------------------
    def _shape_fingerprint(self, dataset : tf.data.Dataset):
        for x, y in dataset.take(1):
            logger.debug(f'X batch shape is: {x.shape}')  
            logger.debug(f'Y batch shape is: {y.shape}') 

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def _build_tensor_dataset(self, images, batch_size, buffer_size=tf.data.AUTOTUNE):

        num_samples = len(images) 
        batch_size = self.configuration["training"]["BATCH_SIZE"] if batch_size is None else batch_size

        dataset = tf.data.Dataset.from_tensor_slices(images)                
        dataset = dataset.map(self.generator.load_image, num_parallel_calls=buffer_size)        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        dataset = dataset.shuffle(buffer_size=num_samples)

        return dataset         
      
    #--------------------------------------------------------------------------
    def build_model_dataloader(self, train_data : pd.DataFrame, validation_data : pd.DataFrame, 
                               batch_size=None):            
        
        train_dataset = self._build_tensor_dataset(train_data, batch_size)
        validation_dataset = self._build_tensor_dataset(validation_data, batch_size)      
        self._shape_fingerprint(train_dataset)

        return train_dataset, validation_dataset






   


    