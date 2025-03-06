import json
from os.path import join, abspath 


# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../.."))
PROJECT_DIR = join(ROOT_DIR, 'FEXT')
RSC_PATH = join(PROJECT_DIR, 'resources')
IMG_DATA_PATH = join(RSC_PATH, 'dataset', 'images')
VALIDATION_PATH = join(RSC_PATH, 'validation')
CHECKPOINT_PATH = join(RSC_PATH, 'checkpoints')
ENCODED_PATH = join(RSC_PATH, 'inference')
ENCODED_INPUT_PATH = join(ENCODED_PATH, 'images')
LOGS_PATH = join(RSC_PATH, 'logs')
SETTING_PATH = join(PROJECT_DIR, 'settings')

# [CONFIGURATIONS]
###############################################################################
CONFIG_PATH = join(SETTING_PATH, 'configurations.json')
with open(CONFIG_PATH, 'r') as file:
    CONFIG = json.load(file)


    


