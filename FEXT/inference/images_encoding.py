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
from FEXT.commons.utils.learning.inferencer import FeatureEncoding
from FEXT.commons.constants import CONFIG, ENCODED_INPUT_PATH
from FEXT.commons.logger import logger



###############################################################################


# 1. [EXTRACT FEATURES FROM IMAGES]
#--------------------------------------------------------------------------   
# select a fraction of data for training
images_paths = get_images_path(ENCODED_INPUT_PATH)

# selected and load the pretrained model, then print the summary     
logger.info('Loading specific checkpoint from pretrained models') 
modelserializer = ModelSerializer()    
model, configuration, history = modelserializer.load_pretrained_model()
model.summary(expand_nested=True)    

# extract features from images using the encoder output    
encoder = FeatureEncoding(model, configuration)    
encoder.encoder_images(images_paths)

# pb = ProgressBarWidgets()
# processor = AdsorptionDataProcessing()
# solver = SolverThread()     
    
# [MAIN PAGE]
#--------------------------------------------------------------------------
with ui.row().classes('w-full no-wrap justify-between'):             
    
    with ui.column().classes('w-full p-4'):
        identify_cols = ui.checkbox('Automatically detect columns')
        best_models = ui.checkbox('Identify best models')
        
    with ui.column().classes('w-full p-4'):
        max_iterations = ui.number("Max iterations", value=1000) 

    with ui.column().classes('w-full p-4'):
        stats = ui.markdown(content="Statistics will be displayed here.").style(
            'padding: 10px; width: 400px; text-align: left;')                           

ui.separator()

# [BUTTONS ROW]
#----------------------------------------------------------------------
with ui.row().classes('w-full no-wrap justify-between'):                         

    with ui.column().classes('w-full p-4'):                
        ui.button('Process data', on_click=lambda : [processor.preprocess_dataset(identify_cols.value),
                                                        stats.set_content(processor.stats)]) 
    with ui.column().classes('w-full p-4'):     
        data_fitting_button = ui.button('Data fitting', on_click=lambda : [solver.start_data_fitting_thread(
                                                                            processor.processed_data,
                                                                            processor.experiment_col,
                                                                            processor.pressure_col,
                                                                            processor.uptake_col,
                                                                            models_widgets.model_states,
                                                                            max_iterations.value,
                                                                            best_models.value)])
        

# [PROGRESS BAR]
#----------------------------------------------------------------------
with ui.row().classes('w-full no-wrap justify-between'):              
    progress_bar = pb.build_progress_bar()

# [TIMER TO UPDATE PROGRESS BAR]
ui.timer(0.5, lambda: pb.update_progress_bar(progress_bar, solver))







ui.run()




