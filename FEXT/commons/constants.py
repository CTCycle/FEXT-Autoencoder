import json
from os.path import join, dirname, abspath 

PROJECT_DIR = dirname(dirname(abspath(__file__)))
RSC_PATH = join(PROJECT_DIR, 'resources')
IMG_DATA_PATH = join(RSC_PATH, 'dataset')
RESULTS_PATH = join(RSC_PATH, 'validation')
CHECKPOINT_PATH = join(RSC_PATH, 'checkpoints')
ENCODED_INPUT_PATH = join(RSC_PATH, 'extraction', 'input images')
ENCODED_OUTPUT_PATH = join(RSC_PATH, 'extraction', 'image features')
LOGS_PATH = join(PROJECT_DIR, 'resources', 'logs')

CONFIG_PATH = join(PROJECT_DIR, 'settings', 'configurations.json')
with open(CONFIG_PATH, 'r') as file:
    CONFIG = json.load(file)

    


