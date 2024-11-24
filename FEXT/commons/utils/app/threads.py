import threading

from FEXT.commons.utils.learning.inference import ImagesEncoding
from FEXT.commons.constants import CONFIG, ENCODED_INPUT_PATH
from FEXT.commons.logger import logger



                
# [FITTING FUNCTION]
###############################################################################
class InferenceThread:

   def __init__(self):              
      
      self.progress = 0
      self.total = 0

   #---------------------------------------------------------------------------
   def start_inference_thread(self, model, configuration):
          
      threading.Thread(target=self.run_inference, args=(model, configuration), daemon=True).start()

   #---------------------------------------------------------------------------
   def run_inference(self, model, configuration):
      
      self.img_encoder = ImagesEncoding(model, configuration)  

      def progress_callback(current, total):
         self.progress = current
         self.total = total

      inference_results = self.img_encoder.encode_images(model, progress_callback=progress_callback)

      

