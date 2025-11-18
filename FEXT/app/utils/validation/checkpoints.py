from __future__ import annotations

import os
import random
import re
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D as MplAxes3D

from FEXT.app.client.workers import check_thread_status, update_progress_callback
from FEXT.app.utils.constants import CHECKPOINT_PATH, EVALUATION_PATH
from FEXT.app.utils.learning.callbacks import LearningInterruptCallback
from FEXT.app.utils.logger import logger
from FEXT.app.utils.repository.serializer import DataSerializer, ModelSerializer
from FEXT.app.utils.services.loader import ImageDataLoader


# [LOAD MODEL]
################################################################################
class ModelEvaluationSummary:
    def __init__(
        self, configuration: dict[str, Any], model: Model | None = None
    ) -> None:
        self.serializer = DataSerializer()
        self.modser = ModelSerializer()
        self.model = model
        self.configuration = configuration

    # --------------------------------------------------------------------------
    def scan_checkpoint_folder(self) -> list[str]:
        model_paths = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():
                pretrained_model_path = os.path.join(entry.path, "saved_model.keras")
                if os.path.isfile(pretrained_model_path):
                    model_paths.append(entry.path)

        return model_paths

    # --------------------------------------------------------------------------
    def get_checkpoints_summary(self, **kwargs) -> pd.DataFrame:
        model_paths = self.scan_checkpoint_folder()
        model_parameters = []
        for i, model_path in enumerate(model_paths):
            _ = self.modser.load_checkpoint(model_path)
            configuration, history = self.modser.load_training_configuration(model_path)
            model_name = os.path.basename(model_path)
            precision = 16 if configuration.get("use_mixed_precision", np.nan) else 32
            has_scheduler = configuration.get("use_scheduler", False)
            scores = history.get("history", {})
            chkp_config = {
                "checkpoint": model_name,
                "sample_size": configuration.get("sample_size", np.nan),
                "validation_size": configuration.get("validation_size", np.nan),
                "seed": configuration.get("train_seed", np.nan),
                "precision": precision,
                "epochs": history.get("epochs", np.nan),
                "batch_size": configuration.get("batch_size", np.nan),
                "split_seed": configuration.get("split_seed", np.nan),
                "image_augmentation": configuration.get("img_augmentation", np.nan),
                "image_height": configuration.get("image_height", np.nan),
                "image_width": configuration.get("image_width", np.nan),
                "image_channels": 1 if configuration.get("use_grayscale", None) else 3,
                "jit_compile": configuration.get("jit_compile", np.nan),
                "has_tensorboard_logs": configuration.get("use_tensorboard", np.nan),
                "initial_LR": configuration.get("initial_LR", np.nan),
                "constant_steps_LR": configuration.get("constant_steps", np.nan)
                if has_scheduler
                else np.nan,
                "decay_steps_LR": configuration.get("decay_steps", np.nan)
                if has_scheduler
                else np.nan,
                "target_LR": configuration.get("target_LR", np.nan)
                if has_scheduler
                else np.nan,
                "initial_neurons": configuration.get("initial_neurons", np.nan),
                "dropout_rate": configuration.get("dropout_rate", np.nan),
                "train_loss": scores.get("loss", [np.nan])[-1],
                "val_loss": scores.get("val_loss", [np.nan])[-1],
                "train_cosine_similarity": scores.get("cosine_similarity", [np.nan])[
                    -1
                ],
                "val_cosine_similarity": scores.get("val_cosine_similarity", [np.nan])[
                    -1
                ],
            }

            model_parameters.append(chkp_config)

            # check for thread status and progress bar update
            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i + 1, len(model_paths), kwargs.get("progress_callback", None)
            )

        dataframe = pd.DataFrame(model_parameters)
        self.serializer.save_checkpoints_summary(dataframe)

        return dataframe

    # -------------------------------------------------------------------------
    def get_evaluation_report(
        self, validation_dataset: tf.data.Dataset, **kwargs
    ) -> None:
        callbacks_list = [LearningInterruptCallback(kwargs.get("worker", None))]
        if self.model:
            validation = self.model.evaluate(
                validation_dataset,
                verbose=1,  # type: ignore
                callbacks=callbacks_list,
            )
            logger.info("Evaluation of pretrained model has been completed")
            logger.info(f"RMSE loss {validation[0]:.3f}")
            logger.info(f"Cosine similarity {validation[1]:.3f}")


# [IMAGE RECONSTRUCTION]
###############################################################################
class ImageReconstruction:
    def __init__(
        self, configuration: dict[str, Any], model: Model, checkpoint_path: str
    ) -> None:
        self.num_images = configuration.get("num_evaluation_images", 6)
        self.img_resolution = 400
        self.file_type = "jpeg"
        self.model = model
        self.configuration = configuration
        # extract checkpoint name and create subfolder in resources/validation
        self.checkpoint = os.path.basename(checkpoint_path)
        self.validation_path = os.path.join(EVALUATION_PATH, self.checkpoint)
        os.makedirs(self.validation_path, exist_ok=True)

    # -------------------------------------------------------------------------
    def save_image(self, fig: Figure, name: str) -> None:
        name = re.sub(r"[^0-9A-Za-z_]", "_", name)
        out_path = os.path.join(self.validation_path, name)
        fig.savefig(out_path, bbox_inches="tight", dpi=self.img_resolution)

    # -------------------------------------------------------------------------
    def get_images(self, data: list[str]) -> list[Any]:
        loader = ImageDataLoader(self.configuration)
        images = [
            loader.load_image(path, as_array=True)
            for path in random.sample(data, self.num_images)
        ]
        norm_images = [loader.image_normalization(img) for img in images]

        return norm_images

    # -------------------------------------------------------------------------
    def visualize_reconstructed_images(
        self, validation_data: list[str], **kwargs
    ) -> Figure:
        val_images = self.get_images(data=validation_data)
        logger.info(
            f"Comparing {self.num_images} reconstructed images from validation dataset"
        )
        fig, axs = plt.subplots(
            nrows=self.num_images, ncols=2, figsize=(4, self.num_images * 2)
        )
        for i, img in enumerate(val_images):
            expanded_img = np.expand_dims(img, axis=0)
            reconstructed_image = self.model.predict(
                expanded_img,
                verbose=0,  # type: ignore
                batch_size=1,
            )[0]

            real = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            pred = np.clip(reconstructed_image * 255.0, 0, 255).astype(np.uint8)
            axs[i, 0].imshow(real)
            axs[i, 0].set_title("Original Picture" if i == 0 else "")
            axs[i, 0].axis("off")
            axs[i, 1].imshow(pred)
            axs[i, 1].set_title("Reconstructed Picture" if i == 0 else "")
            axs[i, 1].axis("off")

            # check for thread status and progress bar update
            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i + 1, len(val_images), kwargs.get("progress_callback", None)
            )

        plt.tight_layout()
        self.save_image(fig, "images_recostruction.jpeg")
        plt.close()

        return fig


# [EMBEDDINGS VISUALIZATION]
###############################################################################
class EmbeddingsVisualization:
    def __init__(
        self, configuration: dict[str, Any], model: Model, checkpoint_path: str
    ) -> None:
        # build encoder sub-model from the compression layer
        encoder_output = model.get_layer("compression_layer").output
        self.encoder_model = Model(inputs=model.input, outputs=encoder_output)
        self.configuration = configuration

        # extract checkpoint name and create subfolder in resources/validation
        self.checkpoint = os.path.basename(checkpoint_path)
        self.validation_path = os.path.join(EVALUATION_PATH, self.checkpoint)
        os.makedirs(self.validation_path, exist_ok=True)

        self.img_resolution = 300
        self.file_type = "jpeg"

    # -------------------------------------------------------------------------
    def save_image(self, fig: Figure, name: str) -> None:
        name = re.sub(r"[^0-9A-Za-z_]", "_", name)
        out_path = os.path.join(self.validation_path, name)
        fig.savefig(out_path, bbox_inches="tight", dpi=self.img_resolution)

    # -------------------------------------------------------------------------
    def visualize_encoder_embeddings(
        self, validation_dataset: tf.data.Dataset, **kwargs
    ) -> Figure:
        # Extract embeddings from encoder across the validation dataset
        embeddings: list[np.ndarray] = []
        num_batches = 0
        for batch in validation_dataset:
            # dataset yields (input, target)
            if isinstance(batch, tuple) and len(batch) == 2:
                inputs = batch[0]
            else:
                inputs = batch

            batch_emb = self.encoder_model.predict(inputs, verbose=0)  # type: ignore
            # flatten spatial dims if present
            batch_emb = np.reshape(batch_emb, (batch_emb.shape[0], -1))
            embeddings.append(batch_emb)
            num_batches += 1

            check_thread_status(kwargs.get("worker", None))

        if not embeddings:
            fig = plt.figure(figsize=(5, 4))
            plt.title("No embeddings extracted")
            return fig

        X = np.concatenate(embeddings, axis=0)
        # Center the data
        X_mean = X.mean(axis=0, keepdims=True)
        X_centered = X - X_mean
        # PCA via SVD for 3 components
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        comps = Vt[:3]
        X_pca = X_centered @ comps.T

        fig = plt.figure(figsize=(6, 5))
        ax = cast(MplAxes3D, fig.add_subplot(111, projection="3d"))
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], s=8, alpha=0.7)
        ax.set_title("Embeddings PCA (3D)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.tight_layout()

        self.save_image(fig, "embeddings_pca_3d.jpeg")
        plt.close()

        return fig
