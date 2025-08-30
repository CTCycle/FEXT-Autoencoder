from __future__ import annotations

from typing import Any

import numpy as np


# [DATA SPLITTING]
###############################################################################
class TrainValidationSplit:
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.validation_size = configuration.get("validation_size", 42)
        np.random.seed(configuration.get("split_seed", 42))
        self.configuration = configuration

    # -------------------------------------------------------------------------
    def split_train_and_validation(
        self, images_path: list
    ) -> tuple[list[Any], list[Any]]:
        # shuffle the paths list to perform randomic sampling
        np.random.shuffle(images_path)
        # get num of samples in train and validation dataset
        self.train_size = int(len(images_path) * (1.0 - self.validation_size))
        self.val_size = int(len(images_path) * self.validation_size)

        shuffled_indices = np.random.permutation(len(images_path))
        train_indices = shuffled_indices[: self.train_size]
        validation_indices = shuffled_indices[self.train_size :]
        train_data = [images_path[i] for i in train_indices]
        validation_data = [images_path[i] for i in validation_indices]

        return train_data, validation_data
