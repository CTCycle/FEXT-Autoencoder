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
        image_statistics = self.analyzer.calculate_image_statistics(images_paths)  

        return image_statistics  

    #--------------------------------------------------------------------------
    def get_pixel_distribution(self, progress_callback=None): 
        images_paths = self.serializer.get_images_path_from_directory(IMG_PATH)            
        self.analyzer.calculate_pixel_intensity_distribution(
            images_paths, progress_callback=progress_callback)             


    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_success(self, window, message):            
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
        logger.error(exc, tb)
        QMessageBox.critical(window, 'Dataset loading failed!', f"{exc}\n\n{tb}")  

   




###############################################################################
class InferenceEvents:

    def __init__(self, configurations):
        self.configurations = configurations       
        self.headless = configurations.get('headless', False)
        self.ignore_SSL = configurations.get('ignore_SSL', False)
        self.wait_time = configurations.get('wait_time', 0)        

    #--------------------------------------------------------------------------
    def get_drug_names(self):         
        filepath = os.path.join(DATA_PATH, 'drugs_to_search.txt')  
        with open(filepath, 'r') as file:
            drug_list = [x.lower().strip() for x in file.readlines()]

        return drug_list  

    #--------------------------------------------------------------------------
    def search_using_webdriver(self, drug_list=None):
        # initialize webdriver and webscraper
        self.toolkit = WebDriverToolkit(self.headless, self.ignore_SSL) 
        webdriver = self.toolkit.initialize_webdriver()
        webscraper = EMAWebPilot(webdriver, self.wait_time)  
        # check if files downloaded in the past are still present, then remove them
        # create a dictionary of drug names with their initial letter as key    
        file_remover()
        if drug_list is None:
            drug_list = self.get_drug_names()

        grouped_drugs = drug_to_letter_aggregator(drug_list)
        # click on letter page (based on first letter of names group) and then iterate over
        # all drugs in that page (from the list). Download excel reports and rename them automatically         
        webscraper.download_manager(grouped_drugs) 




###############################################################################
class TrainingEvents:

    def __init__(self, configurations):
        self.configurations = configurations       
        self.headless = configurations.get('headless', False)
        self.ignore_SSL = configurations.get('ignore_SSL', False)
        self.wait_time = configurations.get('wait_time', 0)        

    #--------------------------------------------------------------------------
    def get_drug_names(self):         
        filepath = os.path.join(DATA_PATH, 'drugs_to_search.txt')  
        with open(filepath, 'r') as file:
            drug_list = [x.lower().strip() for x in file.readlines()]

        return drug_list  

    #--------------------------------------------------------------------------
    def search_using_webdriver(self, drug_list=None):
        # initialize webdriver and webscraper
        self.toolkit = WebDriverToolkit(self.headless, self.ignore_SSL) 
        webdriver = self.toolkit.initialize_webdriver()
        webscraper = EMAWebPilot(webdriver, self.wait_time)  
        # check if files downloaded in the past are still present, then remove them
        # create a dictionary of drug names with their initial letter as key    
        file_remover()
        if drug_list is None:
            drug_list = self.get_drug_names()

        grouped_drugs = drug_to_letter_aggregator(drug_list)
        # click on letter page (based on first letter of names group) and then iterate over
        # all drugs in that page (from the list). Download excel reports and rename them automatically         
        webscraper.download_manager(grouped_drugs) 

