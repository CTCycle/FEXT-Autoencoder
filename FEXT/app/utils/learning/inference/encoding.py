import os

import numpy as np
from tqdm import tqdm
from keras.utils import set_random_seed
from keras import Model

from FEXT.app.client.workers import check_thread_status, update_progress_callback
from FEXT.app.utils.data.loader import ImageDataLoader
from FEXT.app.constants import INFERENCE_PATH
from FEXT.app.logger import logger


# [INFERENCE]
###############################################################################
class ImageEncoding:
    def __init__(self, model : Model, configuration : dict, checkpoint_path : str):
        set_random_seed(configuration.get("train_seed", 42))
        self.checkpoint = os.path.basename(checkpoint_path)
        self.configuration = configuration
        self.model = model
        # isolate the encoder submodel from the autoencoder model
        encoder_output = model.get_layer("compression_layer").output
        self.encoder_model = Model(inputs=model.input, outputs=encoder_output)

    #-------------------------------------------------------------------------
    def encode_img_features(self, images_paths, **kwargs):
        dataloader = ImageDataLoader(self.configuration, shuffle=False)
        features = {}
        for i, pt in enumerate(
            tqdm(images_paths, desc="Encoding images", total=len(images_paths))
        ):
            image_name = os.path.basename(pt)
            try:
                image = dataloader.load_image(pt, as_array=True)
                image = np.expand_dims(image, axis=0)
                extracted_features = self.encoder_model.predict(image, verbose=0)
                features[pt] = extracted_features
            except Exception as e:
                features[pt] = f"Error during encoding: {str(e)}"
                logger.error(f"Could not encode image {image_name}: {str(e)}")

            # check for worker thread status and update progress callback
            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i + 1, len(images_paths), kwargs.get("progress_callback", None)
            )

        # combine extracted features with images name and save them in numpy arrays
        structured_data = np.array(
            [(image, features[image]) for image in features], dtype=object
        )
        file_loc = os.path.join(INFERENCE_PATH, f"encoded_img_{self.checkpoint}.npy")
        np.save(file_loc, structured_data)

        return features
