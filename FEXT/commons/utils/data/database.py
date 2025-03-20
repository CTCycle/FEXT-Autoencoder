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
        

    