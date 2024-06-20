from os.path import join, dirname, abspath 

PROJECT_DIR = dirname(dirname(abspath(__file__)))
IMG_DATA_PATH = join(PROJECT_DIR, 'data', 'images')
VAL_PATH = join(PROJECT_DIR, 'data', 'validation')
CHECKPOINT_PATH = join(PROJECT_DIR, 'training', 'checkpoints')
ENCODED_INPUT_PATH = join(PROJECT_DIR, 'inference', 'images')
ENCODED_OUTPUT_PATH = join(PROJECT_DIR, 'inference', 'extracted_features')


