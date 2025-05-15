from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_agg import FigureCanvasAgg

from FEXT.commons.utils.data.serializer import DataSerializer
from FEXT.commons.utils.validation.images import ImageAnalysis
from FEXT.commons.constants import DATA_PATH, IMG_PATH
from FEXT.commons.logger import logger





###############################################################################
class ValidationEvents:

    def __init__(self, configurations):        
        self.serializer = DataSerializer(configurations)   
        self.analyzer = ImageAnalysis(configurations)     
        self.configurations = configurations               

    #--------------------------------------------------------------------------
    def compute_dataset_statistics(self, progress_callback=None):          
        images_paths = self.serializer.get_images_path_from_directory(IMG_PATH)  
        logger.info(f'The image dataset is composed of {len(images_paths)} images')        
        image_statistics = self.analyzer.calculate_image_statistics(
            images_paths, progress_callback=progress_callback)  

        return image_statistics  

    #--------------------------------------------------------------------------
    def get_pixel_distribution(self, progress_callback=None): 
        images_paths = self.serializer.get_images_path_from_directory(IMG_PATH)            
        distribution = self.analyzer.calculate_pixel_intensity_distribution(
            images_paths, progress_callback=progress_callback)

        return distribution

     #--------------------------------------------------------------------------
    def convert_fig_to_qpixmap(self, fig):    
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        # get the size in pixels and initialize raw RGBA buffer
        width, height = canvas.get_width_height()        
        buf = canvas.buffer_rgba()
        # construct a QImage pointing at that memory (no PNG decoding)
        qimg = QImage(buf, width, height, QImage.Format_RGBA8888)

        return QPixmap.fromImage(qimg)

    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_success(self, window, message):
        # send message to status bar
        window.statusBar().showMessage(message)
    
    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        QMessageBox.critical(window, 'Something went wrong!', f"{exc}\n\n{tb}")  

   


