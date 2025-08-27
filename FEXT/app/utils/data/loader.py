import cv2
import numpy as np
import tensorflow as tf


# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
class ImageDataLoader:
    def __init__(self, configuration: dict, shuffle=True):
        self.image_height = configuration.get("image_height", 256)
        self.image_width = configuration.get("image_width", 256)
        self.channels = 1 if configuration.get("use_grayscale", False) else 3
        self.img_shape = (self.image_height, self.image_width, self.channels)
        self.augmentation = configuration.get("use_img_augmentation", False)
        self.batch_size = configuration.get("batch_size", 32)
        self.inference_batch_size = configuration.get("inference_batch_size", 32)
        self.shuffle_samples = configuration.get("shuffle_size", 1024)
        self.buffer_size = tf.data.AUTOTUNE
        self.color_encoding = (
            cv2.COLOR_BGR2RGB if self.channels == 3 else cv2.COLOR_BGR2GRAY
        )
        self.configuration = configuration
        self.shuffle = shuffle

    # load and preprocess a single image
    #-------------------------------------------------------------------------
    def load_image(self, path, as_array=False):
        if as_array:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, self.color_encoding)
            image = np.asarray(cv2.resize(image, self.img_shape[:-1]), dtype=np.float32)
        else:
            image = tf.io.read_file(path)
            image = tf.image.decode_image(
                image, channels=self.channels, expand_animations=False
            )
            image = tf.image.resize(image, self.img_shape[:-1])

        return image

    # load and preprocess a single image
    #-------------------------------------------------------------------------
    def load_image_for_training(self, path):
        rgb_image = self.load_image(path)
        rgb_image = self.image_normalization(rgb_image)
        rgb_image = (
            self.image_augmentation(rgb_image) if self.augmentation else rgb_image
        )

        return rgb_image, rgb_image

    # load and preprocess a single image
    #-------------------------------------------------------------------------
    def load_image_for_inference(self, path):
        rgb_image = self.load_image(path)
        rgb_image = self.image_normalization(rgb_image)

        return rgb_image

    # define method perform data augmentation
    #-------------------------------------------------------------------------
    def image_normalization(self, image):
        normalized_image = image / 255.0

        return normalized_image

    # define method perform data augmentation
    #-------------------------------------------------------------------------
    def image_augmentation(self, image):
        # perform random image augmentations such as flip, brightness, contrast
        augmentations = {
            "flip_left_right": (lambda img: tf.image.random_flip_left_right(img), 0.5),
            "flip_up_down": (lambda img: tf.image.random_flip_up_down(img), 0.5),
            "brightness": (
                lambda img: tf.image.random_brightness(img, max_delta=0.2),
                0.25,
            ),
            "contrast": (
                lambda img: tf.image.random_contrast(img, lower=0.7, upper=1.3),
                0.35,
            ),
        }

        for _, (func, prob) in augmentations.items():
            if np.random.rand() <= prob:
                image = func(image)

        return image

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #-------------------------------------------------------------------------
    def build_training_dataloader(
        self, images, batch_size=None, buffer_size=tf.data.AUTOTUNE
    ):
        batch_size = self.batch_size if batch_size is None else batch_size
        dataset = tf.data.Dataset.from_tensor_slices(images)
        dataset = dataset.map(
            self.load_image_for_training, num_parallel_calls=buffer_size
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        dataset = (
            dataset.shuffle(buffer_size=self.shuffle_samples)
            if self.shuffle
            else dataset
        )

        return dataset

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #-------------------------------------------------------------------------
    def build_inference_dataloader(
        self, images, batch_size=None, buffer_size=tf.data.AUTOTUNE
    ):
        batch_size = self.inference_batch_size if batch_size is None else batch_size
        dataset = tf.data.Dataset.from_tensor_slices(images)
        dataset = dataset.map(
            self.load_image_for_inference, num_parallel_calls=buffer_size
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)

        return dataset
