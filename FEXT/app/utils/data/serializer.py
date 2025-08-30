from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Union

import pandas as pd
from keras import Model
from keras.models import load_model
from keras.utils import plot_model

from FEXT.app.constants import CHECKPOINT_PATH
from FEXT.app.logger import logger
from FEXT.app.utils.data.database import database
from FEXT.app.utils.learning.training.scheduler import LinearDecayLRScheduler


# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:
    def __init__(self) -> None:
        self.img_shape = (128, 128, 3)
        self.num_channels = self.img_shape[-1]
        self.valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    # -------------------------------------------------------------------------
    def get_img_path_from_directory(
        self, path: str, sample_size: float = 1.0
    ) -> list[str]:
        logger.debug(f"Valid extensions are: {self.valid_extensions}")
        images_path = []
        for root, _, files in os.walk(path):
            if sample_size < 1.0:
                files = files[: int(sample_size * len(files))]
            for file in files:
                if os.path.splitext(file)[1].lower() in self.valid_extensions:
                    images_path.append(os.path.join(root, file))

        return images_path

    # -------------------------------------------------------------------------
    def save_images_statistics(self, data: pd.DataFrame) -> None:
        database.upsert_into_database(data, "IMAGE_STATISTICS")

    # -------------------------------------------------------------------------
    def save_images_exposure_metrics(self, data: pd.DataFrame) -> None:
        database.upsert_into_database(data, "IMAGE_EXPOSURE")

    # -------------------------------------------------------------------------
    def save_images_entropy_metrics(self, data: pd.DataFrame) -> None:
        database.upsert_into_database(data, "IMAGE_ENTROPY")

    # -------------------------------------------------------------------------
    def save_images_sharpness_metrics(self, data: pd.DataFrame) -> None:
        database.upsert_into_database(data, "IMAGE_SHARPNESS")

    # -------------------------------------------------------------------------
    def save_images_edges_metrics(self, data: pd.DataFrame) -> None:
        database.upsert_into_database(data, "IMAGE_EDGES")

    # -------------------------------------------------------------------------
    def save_images_colorimetry(self, data: pd.DataFrame) -> None:
        database.upsert_into_database(data, "IMAGE_COLORIMETRY")

    # -------------------------------------------------------------------------
    def save_images_texture_metric(self, data: pd.DataFrame) -> None:
        database.upsert_into_database(data, "IMAGE_TEXTURE_LBP")

    # -------------------------------------------------------------------------
    def save_checkpoints_summary(self, data: pd.DataFrame) -> None:
        database.upsert_into_database(data, "CHECKPOINTS_SUMMARY")


# [MODEL SERIALIZATION]
###############################################################################
class ModelSerializer:
    def __init__(self) -> None:
        pass

    # function to create a folder where to save model checkpoints
    # -------------------------------------------------------------------------
    def create_checkpoint_folder(self, model_name: str | None = None) -> str:
        today_datetime = datetime.now().strftime("%Y%m%dT%H%M%S")
        checkpoint_path = os.path.join(
            CHECKPOINT_PATH, f"{model_name}_{today_datetime}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(checkpoint_path, "configuration"), exist_ok=True)
        logger.debug(f"Created checkpoint folder at {checkpoint_path}")

        return checkpoint_path

    # -------------------------------------------------------------------------
    def save_pretrained_model(self, model: Model, path) -> None:
        model_files_path = os.path.join(path, "saved_model.keras")
        model.save(model_files_path)
        logger.info(
            f"Training session is over. Model {os.path.basename(path)} has been saved"
        )

    # -------------------------------------------------------------------------
    def save_training_configuration(
        self, path, history, configuration: dict[str, Any]
    ) -> None:
        config_path = os.path.join(path, "configuration", "configuration.json")
        history_path = os.path.join(path, "configuration", "session_history.json")

        # Save training and model configuration
        with open(config_path, "w") as f:
            json.dump(configuration, f)

        # Save session history
        with open(history_path, "w") as f:
            json.dump(history, f)

    # -------------------------------------------------------------------------
    def load_training_configuration(self, path: str) -> tuple[dict, dict]:
        config_path = os.path.join(path, "configuration", "configuration.json")
        history_path = os.path.join(path, "configuration", "session_history.json")

        with open(config_path) as f:
            configuration = json.load(f)

        with open(history_path) as f:
            history = json.load(f)

        return configuration, history

    # -------------------------------------------------------------------------
    def scan_checkpoints_folder(self) -> list[str]:
        model_folders = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():
                # Check if the folder contains at least one .keras file
                has_keras = any(
                    f.name.endswith(".keras") and f.is_file()
                    for f in os.scandir(entry.path)
                )
                if has_keras:
                    model_folders.append(entry.name)

        return model_folders

    # -------------------------------------------------------------------------
    def save_model_plot(self, model: Model, path: str) -> None:
        try:
            plot_path = os.path.join(path, "model_layout.png")
            plot_model(
                model,
                to_file=plot_path,
                show_shapes=True,
                show_layer_names=True,
                show_layer_activations=True,
                expand_nested=True,
                rankdir="TB",
                dpi=400,
            )
            logger.debug(f"Model architecture plot generated as {plot_path}")
        except (OSError, FileNotFoundError, ImportError):
            logger.warning(
                "Could not generate model architecture plot (graphviz/pydot not correctly installed)"
            )

    # -------------------------------------------------------------------------
    def load_checkpoint(
        self, checkpoint: str
    ) -> tuple[Union[Model, Any], dict, dict, str]:
        # effectively load the model using keras builtin method
        # load configuration data from .json file in checkpoint folder
        custom_objects = {"LinearDecayLRScheduler": LinearDecayLRScheduler}
        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint)
        model_path = os.path.join(checkpoint_path, "saved_model.keras")
        model = load_model(model_path, custom_objects=custom_objects)
        configuration, session = self.load_training_configuration(checkpoint_path)

        return model, configuration, session, checkpoint_path
