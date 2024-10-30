import json
from os.path import join, dirname, abspath 


# [PATHS]
###############################################################################
PROJECT_DIR = dirname(dirname(abspath(__file__)))
RSC_PATH = join(PROJECT_DIR, 'resources')
IMG_DATA_PATH = join(RSC_PATH, 'dataset')
RESULTS_PATH = join(RSC_PATH, 'validation')
CHECKPOINT_PATH = join(RSC_PATH, 'checkpoints')
ENCODED_PATH = join(RSC_PATH, 'encoding')
ENCODED_INPUT_PATH = join(ENCODED_PATH, 'input images')
LOGS_PATH = join(RSC_PATH, 'logs')
SETTING_PATH = join(PROJECT_DIR, 'settings')

# [FILENAMES]
###############################################################################
# add filenames here

# [CONFIGURATIONS]
###############################################################################
CONFIG_PATH = join(SETTING_PATH, 'app_configurations.json')
with open(CONFIG_PATH, 'r') as file:
    CONFIG = json.load(file)


    


