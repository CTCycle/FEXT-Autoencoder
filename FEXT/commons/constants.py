import json
from os.path import join, dirname, abspath 

PROJECT_DIR = dirname(dirname(abspath(__file__)))
RSC_PATH = join(PROJECT_DIR, 'resources')
IMG_DATA_PATH = join(RSC_PATH, 'dataset')
RESULTS_PATH = join(RSC_PATH, 'results')
CHECKPOINT_PATH = join(RSC_PATH, 'checkpoints')
ENCODED_INPUT_PATH = join(RSC_PATH, 'encoding', 'input_images')
ENCODED_OUTPUT_PATH = join(RSC_PATH, 'encoding', 'encoder_output')
LOGS_PATH = join(PROJECT_DIR, 'resources', 'logs')

CONFIG_PATH = join(PROJECT_DIR, 'configurations.json')
with open(CONFIG_PATH, 'r') as file:
    CONFIG = json.load(file)


