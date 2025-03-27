import os
import sqlite3
import pandas as pd

from FEXT.commons.constants import DATA_PATH
from FEXT.commons.logger import logger

# [DATABASE]
###############################################################################
class FEXTDatabase:

    def __init__(self, configuration):             
        self.db_path = os.path.join(DATA_PATH, 'FEXT_database.db')               
        self.configuration = configuration 
        self.initialize_database()  

    #--------------------------------------------------------------------------       
    def initialize_database(self):        
        # Connect to the SQLite database and create the database if does not exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()        
        
        create_image_statistics_table = '''
        CREATE TABLE IF NOT EXISTS IMAGE_STATISTICS (
            index INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            height INTEGER,
            width INTEGER,
            mean REAL,
            median REAL,
            std REAL,
            min REAL,
            max REAL,
            pixel_range REAL,
            noise_std REAL,
            noise_ratio REAL
        );
        '''         
    
        create_checkpoints_summary_table = '''
        CREATE TABLE IF NOT EXISTS CHECKPOINTS_SUMMARY (
            index INTEGER PRIMARY KEY AUTOINCREMENT,
            checkpoint_name TEXT,
            sample_size REAL,
            validation_size REAL,
            seed INTEGER,
            precision_bits INTEGER,
            epochs INTEGER,
            additional_epochs INTEGER,
            batch_size INTEGER,
            split_seed INTEGER,
            image_augmentation TEXT,
            image_height INTEGER,
            image_width INTEGER,
            image_channels INTEGER,
            jit_compile TEXT,
            jit_backend TEXT,
            device TEXT,
            device_id TEXT,
            number_of_processors INTEGER,
            use_tensorboard TEXT,
            lr_scheduler_initial_lr REAL,
            lr_scheduler_constant_steps REAL,
            lr_scheduler_decay_steps REAL
        );
        '''
        
        cursor.execute(create_image_statistics_table)   
        cursor.execute(create_checkpoints_summary_table)    

        conn.commit()
        conn.close()       

    #--------------------------------------------------------------------------
    def save_image_statistics(self, data : pd.DataFrame): 
        # connect to sqlite database and save the preprocessed data as table
        conn = sqlite3.connect(self.db_path)         
        data.to_sql('IMAGE_STATISTICS', conn, if_exists='replace')
        conn.commit()
        conn.close() 

    #--------------------------------------------------------------------------
    def save_checkpoints_summary(self, data : pd.DataFrame): 
        # connect to sqlite database and save the preprocessed data as table
        conn = sqlite3.connect(self.db_path)         
        data.to_sql('CHECKPOINTS_SUMMARY', conn, if_exists='replace')
        conn.commit()
        conn.close() 
        

    