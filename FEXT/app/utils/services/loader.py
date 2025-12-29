from __future__ import annotations

from typing import Any, Literal, overload

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info


# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
class ImageDataset(Dataset):
    def __init__(
        self,
        images: list[str],
        loader: "ImageDataLoader",
        for_training: bool,
    ) -> None:
        self.images = list(images)
        self.loader = loader
        self.for_training = for_training

    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.images)

    # -------------------------------------------------------------------------
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        path = self.images[index]
        if self.for_training:
            return self.loader.load_image_for_training(path)
        return self.loader.load_image_for_inference(path)


###############################################################################
def initialize_worker(worker_id: int) -> None:
    worker_info = get_worker_info()
    if worker_info is None:
        return

    dataset = worker_info.dataset
    if isinstance(dataset, ImageDataset):
        dataset.loader.reset_rng(worker_id)


###############################################################################
class ImageDataLoader:
    def __init__(self, configuration: dict[str, Any], shuffle: bool = True) -> None:
        self.image_height = configuration.get("image_height", 256)
        self.image_width = configuration.get("image_width", 256)
        self.channels = 1 if configuration.get("use_grayscale", False) else 3
        self.img_shape = (self.image_height, self.image_width, self.channels)
        self.augmentation = configuration.get("use_img_augmentation", False)
        self.batch_size: int = configuration.get("batch_size", 32)
        self.inference_batch_size: int = configuration.get("inference_batch_size", 32)
        self.shuffle_samples = configuration.get("shuffle_size", 1024)
        self.num_workers: int = configuration.get("num_workers", 0)
        self.prefetch_factor: int = configuration.get("prefetch_factor", 2)
        self.use_pinned_memory: bool = configuration.get("use_device_GPU", False)
        self.color_encoding = (
            cv2.COLOR_BGR2RGB if self.channels == 3 else cv2.COLOR_BGR2GRAY
        )
        self.configuration = configuration
        self.shuffle = shuffle
        self.rng_seed = configuration.get("rng_seed", configuration.get("seed", 42))
        self.rng = np.random.default_rng(self.rng_seed)

    # load and preprocess a single image
    # -------------------------------------------------------------------------
    @overload
    def load_image(
        self, path: str, as_array: Literal[False] = False
    ) -> torch.Tensor: ...

    @overload
    def load_image(self, path: str, as_array: Literal[True]) -> np.ndarray: ...

    def load_image(self, path: str, as_array: bool = False) -> np.ndarray | torch.Tensor:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, self.color_encoding)
        image = cv2.resize(image, self.img_shape[:-1])
        if self.channels == 1 and image.ndim == 2:
            image = image[:, :, None]
        image = np.asarray(image, dtype=np.float32)
        if as_array:
            return image

        image = np.ascontiguousarray(image)
        return torch.from_numpy(image)

    # load and preprocess a single image
    # -------------------------------------------------------------------------
    def load_image_for_training(
        self, path: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_image = self.load_image(path)
        rgb_image = self.image_normalization(rgb_image)
        rgb_image = (
            self.image_augmentation(rgb_image) if self.augmentation else rgb_image
        )

        return rgb_image, rgb_image

    # load and preprocess a single image
    # -------------------------------------------------------------------------
    def load_image_for_inference(self, path: str) -> torch.Tensor:
        rgb_image = self.load_image(path)
        rgb_image = self.image_normalization(rgb_image)

        return rgb_image

    # define method perform data augmentation
    # -------------------------------------------------------------------------
    @overload
    def image_normalization(self, image: torch.Tensor) -> torch.Tensor: ...

    @overload
    def image_normalization(self, image: np.ndarray) -> np.ndarray: ...

    def image_normalization(
        self, image: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        normalized_image = image / 255.0

        return normalized_image

    # define method perform data augmentation
    # -------------------------------------------------------------------------
    @overload
    def image_augmentation(self, image: torch.Tensor) -> torch.Tensor: ...

    @overload
    def image_augmentation(self, image: np.ndarray) -> np.ndarray: ...

    def image_augmentation(
        self, image: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if self.rng.random() <= 0.5:
            image = self.flip_left_right(image)
        if self.rng.random() <= 0.5:
            image = self.flip_up_down(image)
        if self.rng.random() <= 0.25:
            image = self.adjust_brightness(image, max_delta=0.2)
        if self.rng.random() <= 0.35:
            image = self.adjust_contrast(image, lower=0.7, upper=1.3)

        return image

    # -------------------------------------------------------------------------
    def flip_left_right(
        self, image: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if isinstance(image, torch.Tensor):
            return torch.flip(image, dims=[1])
        return np.flip(image, axis=1)

    # -------------------------------------------------------------------------
    def flip_up_down(
        self, image: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if isinstance(image, torch.Tensor):
            return torch.flip(image, dims=[0])
        return np.flip(image, axis=0)

    # -------------------------------------------------------------------------
    def adjust_brightness(
        self, image: np.ndarray | torch.Tensor, max_delta: float
    ) -> np.ndarray | torch.Tensor:
        delta = float(self.rng.uniform(-max_delta, max_delta))
        if isinstance(image, torch.Tensor):
            return torch.clamp(image + delta, 0.0, 1.0)
        return np.clip(image + delta, 0.0, 1.0)

    # -------------------------------------------------------------------------
    def adjust_contrast(
        self, image: np.ndarray | torch.Tensor, lower: float, upper: float
    ) -> np.ndarray | torch.Tensor:
        contrast = float(self.rng.uniform(lower, upper))
        if isinstance(image, torch.Tensor):
            mean_value = torch.mean(image)
            return torch.clamp((image - mean_value) * contrast + mean_value, 0.0, 1.0)

        mean_value = float(image.mean())
        return np.clip((image - mean_value) * contrast + mean_value, 0.0, 1.0)

    # -------------------------------------------------------------------------
    def reset_rng(self, worker_id: int) -> None:
        self.rng = np.random.default_rng(self.rng_seed + worker_id)

    # effectively build the torch dataloader and apply preprocessing, batching and prefetching
    # -------------------------------------------------------------------------
    def build_training_dataloader(
        self, images, batch_size: int | None = None, num_workers: int | None = None
    ) -> DataLoader:
        batch_size = self.batch_size if batch_size is None else batch_size
        num_workers = self.num_workers if num_workers is None else num_workers
        dataset = ImageDataset(list(images), self, for_training=True)
        generator = torch.Generator()
        generator.manual_seed(self.rng_seed)
        loader_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "num_workers": num_workers,
            "pin_memory": self.use_pinned_memory,
            "persistent_workers": num_workers > 0,
            "worker_init_fn": initialize_worker,
            "generator": generator,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(dataset, **loader_kwargs)

    # effectively build the torch dataloader and apply preprocessing, batching and prefetching
    # -------------------------------------------------------------------------
    def build_inference_dataloader(
        self, images, batch_size: int | None = None, num_workers: int | None = None
    ) -> DataLoader:
        batch_size = self.inference_batch_size if batch_size is None else batch_size
        num_workers = self.num_workers if num_workers is None else num_workers
        dataset = ImageDataset(list(images), self, for_training=False)
        loader_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": self.use_pinned_memory,
            "persistent_workers": num_workers > 0,
            "worker_init_fn": initialize_worker,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(dataset, **loader_kwargs)
