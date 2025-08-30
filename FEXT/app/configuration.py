from __future__ import annotations

import json
import os
from typing import Any

from FEXT.app.constants import CONFIG_PATH


###############################################################################
class Configuration:
    def __init__(self) -> None:
        self.configuration = {
            # Dataset
            "seed": 42,
            "sample_size": 1.0,
            "validation_size": 0.2,
            "image_height": 128,
            "image_width": 128,
            "use_grayscale": False,
            "img_augmentation": False,
            "shuffle_dataset": True,
            "shuffle_size": 256,
            # Model
            "selected_model": "FextAE Redux",
            "dropout_rate": 0.2,
            "jit_compile": False,
            "jit_backend": "inductor",
            # Device
            "use_device_GPU": False,
            "device_id": 0,
            "use_mixed_precision": False,
            "num_workers": 0,
            # Training
            "split_seed": 42,
            "train_seed": 42,
            "train_sample_size": 1.0,
            "epochs": 100,
            "additional_epochs": 10,
            "batch_size": 32,
            "plot_training_metrics": True,
            "use_tensorboard": False,
            "save_checkpoints": False,
            "checkpoints_frequency": 1,
            # Learning rate scheduler
            "use_scheduler": False,
            "initial_LR": 0.001,
            "constant_steps": 1000,
            "decay_steps": 500,
            "target_LR": 0.0001,
            # Inference
            "inference_batch_size": 32,
            "num_evaluation_images": 6,
            # viewer
        }

    # -------------------------------------------------------------------------
    def get_configuration(self) -> dict[str, Any]:
        return self.configuration

    # -------------------------------------------------------------------------
    def update_value(self, key: str, value: bool) -> None:
        self.configuration[key] = value

    # -------------------------------------------------------------------------
    def save_configuration_to_json(self, name: str) -> None:
        full_path = os.path.join(CONFIG_PATH, f"{name}.json")
        with open(full_path, "w") as f:
            json.dump(self.configuration, f, indent=4)

    # -------------------------------------------------------------------------
    def load_configuration_from_json(self, name: str) -> None:
        full_path = os.path.join(CONFIG_PATH, name)
        with open(full_path) as f:
            self.configuration = json.load(f)
