from os.path import join, dirname, abspath 

PROJECT_DIR = dirname(abspath(__file__))
MAIN_DIR = join(PROJECT_DIR, 'main')
IMG_DATA_PATH = join(MAIN_DIR, 'data', 'images')
VAL_PATH = join(MAIN_DIR, 'data', 'validation')
CHECKPOINT_PATH = join(MAIN_DIR, 'training', 'checkpoints')
INFERENCE_INPUT_PATH = join(MAIN_DIR, 'inference', 'images')
INFERENCE_OUTPUT_PATH = join(MAIN_DIR, 'inference')


