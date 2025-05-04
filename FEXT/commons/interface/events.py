import os
import io
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QImage, QPixmap

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
        self.analyzer.calculate_pixel_intensity_distribution(
            images_paths, progress_callback=progress_callback)   

    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_success(self, window, message, popup=False): 
        if popup:                
            QMessageBox.information(
            window, 
            "Task successful",
            message,
            QMessageBox.Ok)

        # send message to status bar
        window.statusBar().showMessage(message)
    
    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        QMessageBox.critical(window, 'Something went wrong!', f"{exc}\n\n{tb}")  

   




###############################################################################
class VisualizationEnvents:

    def __init__(self, configurations):
        self.configurations = configurations             
        self.DPI = 400

    #--------------------------------------------------------------------------
    def visualize_benchmark_results(self, tokenizers):        
        self.visualizer.update_tokenizers_dictionaries(tokenizers)

        figures = []
        self.visualizer.get_vocabulary_report()          
        figures.append(self.visualizer.plot_vocabulary_size())
        figures.extend(self.visualizer.plot_histogram_tokens_length())
        figures.append(self.visualizer.plot_boxplot_tokens_length())
        figures.append(self.visualizer.plot_subwords_vs_words())       

        return figures  
    
    #--------------------------------------------------------------------------
    def convert_fig_to_qpixmap(self, fig):    
        buf = io.bytesIO()
        fig.savefig(buf, format="png", dpi=self.DPI)
        buf.seek(0)
        img_data = buf.read()       
        qimg = QImage.fromData(img_data)

        return QPixmap.fromImage(qimg)
    
    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_success(self, window, message, popup=False): 
        if popup:                
            QMessageBox.information(
            window, 
            "Task successful",
            message,
            QMessageBox.Ok)

        # send message to status bar
        window.statusBar().showMessage(message)
    
    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        QMessageBox.critical(window, 'Something went wrong!', f"{exc}\n\n{tb}")  