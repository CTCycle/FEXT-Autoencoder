import threading

from FEXT.commons.utils.dataloader.serializer import get_images_path
from FEXT.commons.utils.dataloader.serializer import ModelSerializer
from FEXT.commons.utils.learning.inferencer import FeatureEncoding
from FEXT.commons.constants import CONFIG, ENCODED_INPUT_PATH
from FEXT.commons.logger import logger



                
# [FITTING FUNCTION]
###############################################################################
class InferenceThread:

   def __init__(self):  
              
        self.progress = 0
        self.total = 0

   #---------------------------------------------------------------------------
   def start_inference_thread(self):
      
      threading.Thread(target=self.run_inference, args=(), daemon=True).start()

   #---------------------------------------------------------------------------
   def run_inference(self):
      
      def progress_callback(current, total):
         self.progress = current
         self.total = total

         # inference_results = self.solver.bulk_data_fitting(processed_data, experiment_col, pressure_col,
         #                                              uptake_col, model_states, max_iterations,
         #                                              progress_callback=progress_callback)

      