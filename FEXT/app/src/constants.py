from os.path import join, abspath 

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../../.."))
PROJECT_DIR = join(ROOT_DIR, 'FEXT')
RSC_PATH = join(PROJECT_DIR, 'resources')
DATA_PATH = join(RSC_PATH, 'database')
IMG_PATH = join(DATA_PATH, 'dataset')
EVALUATION_PATH = join(DATA_PATH, 'validation')
CHECKPOINT_PATH = join(RSC_PATH, 'checkpoints')
INFERENCE_PATH = join(DATA_PATH, 'inference')
INFERENCE_INPUT_PATH = join(INFERENCE_PATH, 'images')
LOGS_PATH = join(RSC_PATH, 'logs')

# [UI LAYOUT PATH]
###############################################################################
UI_PATH = join(PROJECT_DIR, 'app', 'src', 'window_layout.ui')



    


