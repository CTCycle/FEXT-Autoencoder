from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_agg import FigureCanvasAgg

from FEXT.commons.utils.data.serializer import DataSerializer
from FEXT.commons.utils.validation.images import ImageAnalysis
from FEXT.commons.utils.data.loader import TrainingDataLoader
from FEXT.commons.utils.data.serializer import DataSerializer, ModelSerializer
from FEXT.commons.utils.data.splitting import TrainValidationSplit
from FEXT.commons.utils.learning.training import ModelTraining
from FEXT.commons.utils.learning.autoencoder import FeXTAutoEncoder
from FEXT.commons.utils.validation.reports import log_training_report
from FEXT.commons.constants import DATA_PATH, IMG_PATH
from FEXT.commons.logger import logger



###############################################################################
class ValidationEvents:

    def __init__(self, configuration):        
        self.serializer = DataSerializer(configuration)   
        self.analyzer = ImageAnalysis(configuration)     
        self.configuration = configuration    
        
    #--------------------------------------------------------------------------
    def run_dataset_evaluation_pipeline(self, metrics, progress_callback=None):                  
        images_paths = self.serializer.get_images_path_from_directory(IMG_PATH)  
        logger.info(f'The image dataset is composed of {len(images_paths)} images')
        
        images = []        
        if 'image_stats' in metrics:
            logger.info('Current metric: image dataset statistics')
            image_statistics = self.analyzer.calculate_image_statistics(
                images_paths, progress_callback=progress_callback)
             
        if 'pixels_distribution' in metrics:
            logger.info('Current metric: pixel intensity distribution')
            images.append(self.analyzer.calculate_pixel_intensity_distribution(
                images_paths, progress_callback=progress_callback))       

        return images     

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

   

###############################################################################
class TrainingEvents:

    def __init__(self, configuration):        
        self.serializer = DataSerializer(configuration)   
        self.splitter = TrainValidationSplit(configuration)
        self.builder = TrainingDataLoader(configuration)        
        self.trainer = ModelTraining(configuration)  
        self.autoencoder = FeXTAutoEncoder(configuration) 
        self.modelserializer = ModelSerializer()         
        self.configuration = configuration 

    #--------------------------------------------------------------------------
    def get_available_checkpoints(self):
        return self.modelserializer.scan_checkpoints_folder()
            
    #--------------------------------------------------------------------------
    def run_training_pipeline(self, progress_callback=None):  
        logger.info('Preparing dataset of images based on splitting sizes')  
        sample_size = self.configuration.get("train_sample_size", 1.0)
        images_paths = self.serializer.get_images_path_from_directory(IMG_PATH, sample_size) 
        train_data, validation_data = self.splitter.split_train_and_validation(images_paths)
        
        # create the tf.datasets using the previously initialized generators 
        logger.info('Building model data loaders with prefetching and parallel processing')            
        train_dataset, validation_dataset = self.builder.build_training_dataloader(
            train_data, validation_data)
        
        # set device for training operations based on user configuration
        logger.info('Setting device for training operations based on user configuration')         
        self.trainer.set_device()

        # build the autoencoder model 
        logger.info('Building FeXT AutoEncoder model based on user configuration') 
        checkpoint_path = self.modelserializer.create_checkpoint_folder()          
        model = self.autoencoder.get_model(model_summary=True) 

        # generate training log report and graphviz plot for the model layout         
        log_training_report(train_data, validation_data, self.configuration)        
        self.modelserializer.save_model_plot(model, checkpoint_path) 
        # perform training and save model at the end
        logger.info('Starting FeXT AutoEncoder training') 
        self.trainer.train_model(
            model, train_dataset, validation_dataset, checkpoint_path, 
            progress_callback=progress_callback)
        
    #--------------------------------------------------------------------------
    def resume_training_pipeline(self, selected_checkpoint, progress_callback=None):
        logger.info(f'Loading {selected_checkpoint} checkpoint from pretrained models')         
        model, configuration, session, checkpoint_path = self.modelserializer.load_checkpoint()    
        model.summary(expand_nested=True)  
        
        # set device for training operations based on user configuration
        logger.info('Setting device for training operations based on user configuration')         
        self.trainer.set_device() 

        logger.info('Preparing dataset of images based on splitting sizes')  
        sample_size = self.configuration.get("train_sample_size", 1.0)
        images_paths = self.serializer.get_images_path_from_directory(IMG_PATH, sample_size) 
        train_data, validation_data = self.splitter.split_train_and_validation(images_paths)       

        # create the tf.datasets using the previously initialized generators 
        logger.info('Building model data loaders with prefetching and parallel processing')            
        train_dataset, validation_dataset = self.builder.build_training_dataloader(
            train_data, validation_data)        
                            
        # resume training from pretrained model    
        logger.info('Resuming FeXT AutoEncoder training from checkpoint') 
        self.trainer.train_model(
            model, train_dataset, validation_dataset, checkpoint_path,
            from_checkpoint=True)


        
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

   



