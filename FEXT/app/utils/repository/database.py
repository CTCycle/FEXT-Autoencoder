from __future__ import annotations

import os
from typing import Any

import pandas as pd
import sqlalchemy
from sqlalchemy import Column, Float, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import declarative_base, sessionmaker

from FEXT.app.utils.constants import DATA_PATH
from FEXT.app.utils.logger import logger
from FEXT.app.utils.singleton import singleton

Base = declarative_base()


###############################################################################
class ImageStatistics(Base):
    __tablename__ = "IMAGE_STATISTICS"
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
    __table_args__ = (UniqueConstraint("name"),)


###############################################################################
class ImageExposure(Base):
    __tablename__ = "IMAGE_EXPOSURE"
    name = Column(String, primary_key=True)
    underexposed_pct = Column(Float)
    overexposed_pct = Column(Float)
    midtone_pct = Column(Float)
    mean_gray = Column(Float)
    __table_args__ = (UniqueConstraint("name"),)


###############################################################################
class ImageEntropy(Base):
    __tablename__ = "IMAGE_ENTROPY"
    name = Column(String, primary_key=True)
    entropy_bits = Column(Float)
    normalized_entropy = Column(Float)
    __table_args__ = (UniqueConstraint("name"),)


###############################################################################
class ImageSharpness(Base):
    __tablename__ = "IMAGE_SHARPNESS"
    name = Column(String, primary_key=True)
    var_laplacian = Column(Float)
    tenengrad = Column(Float)
    spectral_highfreq_ratio = Column(Float)
    __table_args__ = (UniqueConstraint("name"),)


###############################################################################
class ImageColorimetry(Base):
    __tablename__ = "IMAGE_COLORIMETRY"
    name = Column(String, primary_key=True)
    colorfulness = Column(Float)
    mean_saturation = Column(Float)
    low_sat_pct = Column(Float)
    high_sat_pct = Column(Float)
    __table_args__ = (UniqueConstraint("name"),)


###############################################################################
class ImageEdge(Base):
    __tablename__ = "IMAGE_EDGES"
    name = Column(String, primary_key=True)
    edge_density = Column(Float)
    mean_gradient = Column(Float)
    __table_args__ = (UniqueConstraint("name"),)


###############################################################################
class ImageTextureLbp(Base):
    __tablename__ = "IMAGE_TEXTURE_LBP"
    name = Column(String, primary_key=True)
    lbp_energy = Column(Float)
    lbp_entropy = Column(Float)
    __table_args__ = (UniqueConstraint("name"),)


###############################################################################
class CheckpointSummary(Base):
    __tablename__ = "CHECKPOINTS_SUMMARY"
    checkpoint = Column(String, primary_key=True)
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
    __table_args__ = (UniqueConstraint("checkpoint"),)


# [DATABASE]
###############################################################################
@singleton
class FEXTDatabase:
    def __init__(self) -> None:
        self.db_path = os.path.join(DATA_PATH, "sqlite.db")
        self.engine = create_engine(
            f"sqlite:///{self.db_path}", echo=False, future=True
        )
        self.Session = sessionmaker(bind=self.engine, future=True)
        self.insert_batch_size = 1000

    # -------------------------------------------------------------------------
    def initialize_database(self) -> None:
        if not os.path.exists(self.db_path):
            Base.metadata.create_all(self.engine)

    # -------------------------------------------------------------------------
    def get_table_class(self, table_name: str) -> Any:
        for cls in Base.__subclasses__():
            if hasattr(cls, "__tablename__") and cls.__tablename__ == table_name:
                return cls
        raise ValueError(f"No table class found for name {table_name}")

    # -------------------------------------------------------------------------
    def upsert_dataframe(self, df: pd.DataFrame, table_cls) -> None:
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
            records = df.to_dict(orient="records")
            for i in range(0, len(records), self.insert_batch_size):
                batch = records[i : i + self.insert_batch_size]
                stmt = insert(table).values(batch)
                # Columns to update on conflict
                update_cols = {
                    c: getattr(stmt.excluded, c)  # type: ignore
                    for c in batch[0]
                    if c not in unique_cols
                }
                stmt = stmt.on_conflict_do_update(
                    index_elements=unique_cols, set_=update_cols
                )
                session.execute(stmt)
            session.commit()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def load_from_database(self, table_name: str) -> pd.DataFrame:
        with self.engine.connect() as conn:
            data = pd.read_sql_table(table_name, conn)

        return data

    # -------------------------------------------------------------------------
    def save_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(sqlalchemy.text(f'DELETE FROM "{table_name}"'))
            df.to_sql(table_name, conn, if_exists="append", index=False)

    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        table_cls = self.get_table_class(table_name)
        self.upsert_dataframe(df, table_cls)

    # -------------------------------------------------------------------------
    def export_all_tables_as_csv(
        self, export_dir: str, chunksize: int | None = None
    ) -> None:
        os.makedirs(export_dir, exist_ok=True)
        with self.engine.connect() as conn:
            for table in Base.metadata.sorted_tables:
                table_name = table.name
                csv_path = os.path.join(export_dir, f"{table_name}.csv")

                # Build a safe SELECT for arbitrary table names (quote with "")
                query = sqlalchemy.text(f'SELECT * FROM "{table_name}"')

                if chunksize:
                    first = True
                    for chunk in pd.read_sql(query, conn, chunksize=chunksize):
                        chunk.to_csv(
                            csv_path,
                            index=False,
                            header=first,
                            mode="w" if first else "a",
                            encoding="utf-8",
                            sep=",",
                        )
                        first = False
                    # If no chunks were returned, still write the header row
                    if first:
                        pd.DataFrame(columns=[c.name for c in table.columns]).to_csv(
                            csv_path, index=False, encoding="utf-8", sep=","
                        )
                else:
                    df = pd.read_sql(query, conn)
                    if df.empty:
                        pd.DataFrame(columns=[c.name for c in table.columns]).to_csv(
                            csv_path, index=False, encoding="utf-8", sep=","
                        )
                    else:
                        df.to_csv(csv_path, index=False, encoding="utf-8", sep=",")

        logger.info(f"All tables exported to CSV at {os.path.abspath(export_dir)}")

    # -------------------------------------------------------------------------
    def delete_all_data(self) -> None:
        with self.engine.begin() as conn:
            for table in reversed(Base.metadata.sorted_tables):
                conn.execute(table.delete())


# -----------------------------------------------------------------------------
database = FEXTDatabase()
