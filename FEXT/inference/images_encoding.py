# [SET ML BACKEND]
import os
from nicegui import ui
os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.dataloader.serializer import ModelSerializer
from FEXT.commons.utils.learning.inference import ImagesEncoding
from FEXT.commons.utils.app.widgets import CheckpointLoadingWidgets, ProgressBarWidgets
from FEXT.commons.utils.app.threads import InferenceThread
from FEXT.commons.constants import CONFIG, ENCODED_INPUT_PATH
from FEXT.commons.logger import logger


###############################################################################
with ui.row().classes('w-full no-wrap justify-between') as page:

    # initialize classes used for backend ops and UI widgets building
    # inference is run using the InferenceThread class for asynchronous run
    pb = ProgressBarWidgets()
    selector = CheckpointLoadingWidgets()
    modelserializer = ModelSerializer()  
    inference = InferenceThread()   
    
    #--------------------------------------------------------------------------           
    with ui.column().classes('w-full'):
        ui.label('Device Selection')
        device = ui.select(['CPU', 'GPU'], value='CPU', label='Select device').classes('w-full')  
        npy_save = ui.checkbox('Save as .npy').classes('mt-2')         
    with ui.column().classes('w-full'):
        selector.build_checkpoints_selector()        

    with ui.column().classes('w-full'):
        ui.label('Model inference images')  
        loader_button = ui.button('Load model', on_click=lambda : [modelserializer.load_checkpoint(selector.selected_checkpoint.value,
                                                                                                   selector.summary.value),
                                                                   modelserializer.load_session_configuration(selector.selected_checkpoint.value)])
        inference_button = ui.button('Encode images', on_click=lambda : [inference.start_inference_thread(modelserializer.model,
                                                                                                          modelserializer.configuration)])

    # [PROGRESS BAR]
    #--------------------------------------------------------------------------
    with ui.row().classes('w-full no-wrap justify-between'):
        progress_bar = pb.build_progress_bar()

    # [TIMER TO UPDATE PROGRESS BAR]
    ui.timer(0.5, lambda: pb.update_progress_bar(progress_bar, inference))



ui.run()


