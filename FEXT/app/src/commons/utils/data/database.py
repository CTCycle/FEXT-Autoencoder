import os
import sqlite3
import pandas as pd

from FEXT.app.src.commons.constants import DATA_PATH
from FEXT.app.src.commons.logger import logger


###############################################################################
class ImageStatisticsTable:

    def __init__(self):
        self.name = 'IMAGE_STATISTICS'
        self.dtypes = {
            'name': 'VARCHAR',
            'height': 'INTEGER',
            'width': 'INTEGER',
            'mean': 'FLOAT',
            'median': 'FLOAT',
            'std': 'FLOAT',
            'min': 'FLOAT',
            'max': 'FLOAT',
            'pixel_range': 'FLOAT',
            'noise_std': 'FLOAT',
            'noise_ratio': 'FLOAT'}
        self.unique_col = 'name'

    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (
            name VARCHAR PRIMARY KEY,
            height INTEGER,
            width INTEGER,
            mean FLOAT,
            median FLOAT,
            std FLOAT,
            min FLOAT,
            max FLOAT,
            pixel_range FLOAT,
            noise_std FLOAT,
            noise_ratio FLOAT
        );
        '''
        cursor.execute(query) 
    
    
###############################################################################
class CheckpointSummaryTable:

    def __init__(self):
        self.name = 'CHECKPOINTS_SUMMARY'
        self.dtypes = {
            'checkpoint_name': 'VARCHAR',
            'sample_size': 'FLOAT',
            'validation_size': 'FLOAT',
            'seed': 'INTEGER',
            'precision_bits': 'INTEGER',
            'epochs': 'INTEGER',
            'additional_epochs': 'INTEGER',
            'batch_size': 'INTEGER',
            'split_seed': 'INTEGER',
            'image_augmentation': 'VARCHAR',
            'image_height': 'INTEGER',
            'image_width': 'INTEGER',
            'image_channels': 'INTEGER',
            'jit_compile': 'VARCHAR',
            'jit_backend': 'VARCHAR',
            'device': 'VARCHAR',
            'device_id': 'VARCHAR',
            'number_of_processors': 'INTEGER',
            'use_tensorboard': 'VARCHAR',
            'lr_scheduler_initial_lr': 'FLOAT',
            'lr_scheduler_constant_steps': 'FLOAT',
            'lr_scheduler_decay_steps': 'FLOAT'}    

    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (            
            checkpoint_name VARCHAR,
            sample_size FLOAT,
            validation_size FLOAT,
            seed INTEGER,
            precision_bits INTEGER,
            epochs INTEGER,
            additional_epochs INTEGER,
            batch_size INTEGER,
            split_seed INTEGER,
            image_augmentation VARCHAR,
            image_height INTEGER,
            image_width INTEGER,
            image_channels INTEGER,
            jit_compile VARCHAR,
            jit_backend VARCHAR,
            device VARCHAR,
            device_id VARCHAR,
            number_of_processors INTEGER,
            use_tensorboard VARCHAR,
            lr_scheduler_initial_lr FLOAT,
            lr_scheduler_constant_steps FLOAT,
            lr_scheduler_decay_steps FLOAT
            );
            '''  
        
        cursor.execute(query)       


# [DATABASE]
###############################################################################
class FEXTDatabase:

    def __init__(self, configuration):             
        self.db_path = os.path.join(DATA_PATH, 'FEXT_database.db')               
        self.configuration = configuration
        self.image_stats = ImageStatisticsTable()
        self.checkpoints_summary = CheckpointSummaryTable()         
        
    #--------------------------------------------------------------------------       
    def initialize_database(self):        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor() 
        self.image_stats.create_table(cursor)  
        self.checkpoints_summary.create_table(cursor)   
        conn.commit()
        conn.close()       

    #--------------------------------------------------------------------------
    def save_image_statistics_table(self, data):        
        conn = sqlite3.connect(self.db_path)         
        data.to_sql(
            self.image_stats.name, conn, if_exists='replace', index=False,
            dtype=self.image_stats.get_dtypes())
        conn.commit()
        conn.close() 

    #--------------------------------------------------------------------------
    def save_checkpoints_summary_table(self, data):         
        conn = sqlite3.connect(self.db_path)         
        data.to_sql(
            self.checkpoints_summary.name, conn, if_exists='replace', index=False,
            dtype=self.checkpoints_summary.get_dtypes())
        conn.commit()
        conn.close() 
        

    