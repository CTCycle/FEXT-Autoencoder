import os
import pandas as pd
import sqlalchemy
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Float, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.dialects.sqlite import insert

from FEXT.app.constants import DATA_PATH
from FEXT.app.logger import logger

Base = declarative_base()


###############################################################################
class ImageStatistics(Base):
    __tablename__ = 'IMAGE_STATISTICS'
    name = Column(String, primary_key=True)
    height = Column(Integer)
    width = Column(Integer)
    mean = Column(Float)
    median = Column(Float)
    std = Column(Float)
    min = Column(Float)
    max = Column(Float)
    pixel_range = Column(Float)
    noise_std = Column(Float)
    noise_ratio = Column(Float)
    __table_args__ = (
        UniqueConstraint('name'),
    )
    
###############################################################################
class CheckpointSummary(Base):
    __tablename__ = 'CHECKPOINTS_SUMMARY'
    checkpoint_name = Column(String, primary_key=True)
    sample_size = Column(Float)
    validation_size = Column(Float)
    seed = Column(Integer)
    precision = Column(Integer)
    epochs = Column(Integer)
    batch_size = Column(Integer)
    split_seed = Column(Integer)
    image_augmentation = Column(String)
    image_height = Column(Integer)
    image_width = Column(Integer)
    image_channels = Column(Integer)
    jit_compile = Column(String) 
    has_tensorboard_logs = Column(String)
    initial_LR = Column(Float)
    constant_steps_LR = Column(Float)
    decay_steps_LR = Column(Float)
    target_LR = Column(Float)
    initial_neurons = Column(Float)
    dropout_rate = Column(Float)
    train_loss = Column(Float)
    val_loss = Column(Float)
    train_cosine_similarity = Column(Float)
    val_cosine_similarity = Column(Float)
    __table_args__ = (
        UniqueConstraint('checkpoint_name'),
    )


# [DATABASE]
###############################################################################
class FEXTDatabase:

    def __init__(self):             
        self.db_path = os.path.join(DATA_PATH, 'FEXT_database.db')  
        self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False, future=True)
        self.Session = sessionmaker(bind=self.engine, future=True)
        self.insert_batch_size = 5000
        
    #--------------------------------------------------------------------------       
    def initialize_database(self):
        Base.metadata.create_all(self.engine)
        
    #--------------------------------------------------------------------------
    def upsert_dataframe(self, df: pd.DataFrame, table_cls):
        table = table_cls.__table__
        session = self.Session()
        try:
            unique_cols = []
            for uc in table.constraints:
                if isinstance(uc, UniqueConstraint):
                    unique_cols = uc.columns.keys()
                    break
            if not unique_cols:
                raise ValueError(f"No unique constraint found for {table_cls.__name__}")

            # Batch insertions for speed
            records = df.to_dict(orient='records')
            for i in range(0, len(records), self.insert_batch_size):
                batch = records[i:i + self.insert_batch_size]
                stmt = insert(table).values(batch)
                # Columns to update on conflict
                update_cols = {c: getattr(stmt.excluded, c) for c in batch[0] if c not in unique_cols}
                stmt = stmt.on_conflict_do_update(
                    index_elements=unique_cols,
                    set_=update_cols
                )
                session.execute(stmt)
                session.commit()
            session.commit()
        finally:
            session.close()

    #--------------------------------------------------------------------------
    def save_image_statistics(self, data : pd.DataFrame):      
        with self.engine.begin() as conn:
            conn.execute(sqlalchemy.text(f"DELETE FROM IMAGE_STATISTICS"))        
        data.to_sql('IMAGE_STATISTICS', self.engine, if_exists='append', index=False) 
        
    #--------------------------------------------------------------------------
    def save_checkpoints_summary(self, data : pd.DataFrame):         
        self.upsert_dataframe(data, CheckpointSummary)
        

    