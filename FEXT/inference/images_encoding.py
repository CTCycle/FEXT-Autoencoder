# [SET ML BACKEND]
import os
from nicegui import ui
os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.dataloader.serializer import get_images_path
from FEXT.commons.utils.dataloader.serializer import ModelSerializer
from FEXT.commons.utils.learning.inferencer import ImagesEncoding
from FEXT.commons.constants import CONFIG, ENCODED_INPUT_PATH
from FEXT.commons.logger import logger

from FEXT.commons.utils.app.widgets import CheckpointLoadingWidgets, ProgressBarWidgets
from FEXT.commons.utils.app.threads import InferenceThread


###############################################################################
with ui.row().classes('w-full no-wrap justify-between') as page:

    pb = ProgressBarWidgets()
    selector = CheckpointLoadingWidgets()
    modelserializer = ModelSerializer()  
    inferencer = InferenceThread()
   

    # [SOLVER SECTION]
    #--------------------------------------------------------------------------           
    with ui.column().classes('w-full'):
        ui.label('Device Selection')
        device = ui.select(['CPU', 'GPU'], label='Select device').classes('w-full')  
        #ui.checkbox('Enable device').classes('mt-2')  
    with ui.column().classes('w-full'):
        selector.build_checkpoints_selector()
        

    with ui.column().classes('w-full'):
        ui.label('Model inference images')  
        inference_button = ui.button('Encode images', on_click=lambda : [solver.start_data_fitting_thread(
                                                                                   processor.processed_data,
                                                                                   processor.experiment_col,
                                                                                   processor.pressure_col,
                                                                                   processor.uptake_col,
                                                                                   models_widgets.model_states,
                                                                                   max_iterations.value,
                                                                                   best_models.value)])

    # [PROGRESS BAR]
    #--------------------------------------------------------------------------
    with ui.row().classes('w-full no-wrap justify-between'):
        progress_bar = pb.build_progress_bar()

    # [TIMER TO UPDATE PROGRESS BAR]
    ui.timer(0.5, lambda: pb.update_progress_bar(progress_bar, inferencer))



ui.run()


