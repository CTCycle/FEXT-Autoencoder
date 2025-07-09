import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QImage, QPixmap

from FEXT.app.src.utils.data.serializer import DataSerializer, ModelSerializer
from FEXT.app.src.utils.data.loader import ImageDataLoader
from FEXT.app.src.utils.data.process import TrainValidationSplit
from FEXT.app.src.utils.learning.device import DeviceConfig
from FEXT.app.src.utils.learning.training.fitting import ModelTraining
from FEXT.app.src.utils.learning.models.autoencoder import FeXTAutoEncoder
from FEXT.app.src.utils.learning.inference.encoding import ImageEncoding
from FEXT.app.src.utils.validation.dataset import ImageAnalysis
from FEXT.app.src.utils.validation.checkpoints import ModelEvaluationSummary, ImageReconstruction
from FEXT.app.src.interface.workers import check_thread_status

from FEXT.app.src.constants import IMG_PATH, INFERENCE_INPUT_PATH
from FEXT.app.src.logger import logger


###############################################################################
class GraphicsHandler:

    def __init__(self): 
        self.image_encoding = cv2.IMREAD_UNCHANGED
        self.gray_scale_encoding = cv2.IMREAD_GRAYSCALE
        self.BGRA_encoding = cv2.COLOR_BGRA2RGBA
        self.BGR_encoding = cv2.COLOR_BGR2RGB

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
    
    #--------------------------------------------------------------------------    
    def load_image_as_pixmap(self, path):  
        img = cv2.imread(path, self.image_encoding)
        if img is None:
            return  

        # Convert to RGB or RGBA as needed
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, self.gray_scale_encoding)
            qimg_format = QImage.Format_RGB888
            channels = 3
        elif img.shape[2] == 4:  # BGRA
            img = cv2.cvtColor(img, self.BGRA_encoding)
            qimg_format = QImage.Format_RGBA8888
            channels = 4
        else:  # BGR
            img = cv2.cvtColor(img, self.BGR_encoding)
            qimg_format = QImage.Format_RGB888
            channels = 3

        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, channels * w, qimg_format)
        return QPixmap.fromImage(qimg)


###############################################################################
class ValidationEvents:

    def __init__(self, database, configuration):        
        self.database = database     
        self.eval_batch_size = configuration.get('eval_batch_size', 32)
        self.configuration = configuration  

    #--------------------------------------------------------------------------
    def load_images_path(self, path, sample_size=1.0):
        serializer = DataSerializer(self.database, self.configuration)             
        images_paths = serializer.get_images_path_from_directory(
            path, sample_size) 
        
        return images_paths 
        
    #--------------------------------------------------------------------------
    def run_dataset_evaluation_pipeline(self, metrics, progress_callback=None, worker=None):
        # initialize data serializer for saving and loading data         
        serializer = DataSerializer(self.database, self.configuration) 
        # get images path from the dataset folder and select a randomized fraction    
        sample_size = self.configuration.get("sample_size", 1.0)
        images_paths = serializer.get_images_path_from_directory(IMG_PATH, sample_size)
        logger.info(f'The image dataset is composed of {len(images_paths)} images')            

        # perform image dataset statistical analysis to retrieve common statistics
        # - pixel mean and standard deviation
        # - noise-to-signal ratio
        # - max and min intensity      
        logger.info('Current metric: image dataset statistics')
        analyzer = ImageAnalysis(self.database, self.configuration) 
        image_statistics = analyzer.calculate_image_statistics(
            images_paths, progress_callback=progress_callback, worker=worker)
        logger.info('Image dataset statistics have been updated in the database')                      

        images = []
        # calculate and plot the pixel intensity histogram  
        if 'pixels_distribution' in metrics:
            logger.info('Current metric: pixel intensity distribution')
            images.append(analyzer.calculate_pixel_intensity_distribution(
                images_paths, progress_callback=progress_callback, worker=worker))
            logger.info('Pixel intensity distribution plot is available in viewer')

        return images 

    #--------------------------------------------------------------------------
    def get_checkpoints_summary(self, progress_callback=None, worker=None): 
        summarizer = ModelEvaluationSummary(self.database, self.configuration)    
        checkpoints_summary = summarizer.get_checkpoints_summary(
            progress_callback=progress_callback, worker=worker) 
        logger.info(f'Checkpoints summary has been created for {checkpoints_summary.shape[0]} models')   
    
    #--------------------------------------------------------------------------
    def run_model_evaluation_pipeline(self, metrics, selected_checkpoint, progress_callback=None, worker=None):       
        logger.info(f'Loading {selected_checkpoint} checkpoint from pretrained models')   
        modser = ModelSerializer()       
        model, train_config, session, checkpoint_path = modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  

        # set device for training operations based on user configuration
        logger.info('Setting device for training operations based on user configuration')                
        device = DeviceConfig(self.configuration)
        device.set_device()  

        # isolate the encoder from the autoencoder model   
        encoder = ImageEncoding(model, train_config, checkpoint_path)
        encoder_model = encoder.encoder_model 

        logger.info('Preparing dataset of images based on splitting sizes')  
        sample_size = train_config.get("train_sample_size", 1.0)
        serializer = DataSerializer(self.database, self.configuration)  
        images_paths = serializer.get_images_path_from_directory(IMG_PATH, sample_size)
        splitter = TrainValidationSplit(train_config) 
        _, validation_images = splitter.split_train_and_validation(images_paths)     

        # create the tf.datasets using the previously initialized generators 
        logger.info('Building model data loaders with prefetching and parallel processing') 
        # use tf.data.Dataset to build the model dataloader with a larger batch size
        # the dataset is built on top of the training and validation data
        loader = ImageDataLoader(train_config, shuffle=False)    
        validation_dataset = loader.build_training_dataloader(validation_images)   

        # check worker status to allow interruption
        check_thread_status(worker)             

        images = []
        if 'evaluation_report' in metrics:
            logger.info('Current metric: model loss and metrics evaluation')
            # evaluate model performance over the training and validation dataset 
            summarizer = ModelEvaluationSummary(self.database, self.configuration)       
            summarizer.get_evaluation_report(model, validation_dataset, worker=worker)              

        if 'image_reconstruction' in metrics:
            logger.info('Current metric: image reconstruction')
            validator = ImageReconstruction(train_config, model, checkpoint_path)      
            images.append(validator.visualize_reconstructed_images(
                validation_images, progress_callback=progress_callback, worker=worker))   
            logger.info('Image reconstruction analysis successfully performed')    

        return images   
         

###############################################################################
class ModelEvents:

    def __init__(self, database, configuration): 
        self.database = database          
        self.configuration = configuration 

    #--------------------------------------------------------------------------
    def get_available_checkpoints(self):
        serializer = ModelSerializer()
        return serializer.scan_checkpoints_folder()
            
    #--------------------------------------------------------------------------
    def run_training_pipeline(self, progress_callback=None, worker=None):  
        logger.info('Preparing dataset of images based on splitting sizes')  
        sample_size = self.configuration.get("train_sample_size", 1.0)
        serializer = DataSerializer(self.database, self.configuration)  
        images_paths = serializer.get_images_path_from_directory(IMG_PATH, sample_size)

        splitter = TrainValidationSplit(self.configuration) 
        train_data, validation_data = splitter.split_train_and_validation(images_paths)
        
        # create the tf.datasets using the previously initialized generators 
        logger.info('Building model data loaders with prefetching and parallel processing')     
        builder = ImageDataLoader(self.configuration)          
        train_dataset = builder.build_training_dataloader(train_data)
        validation_dataset = builder.build_training_dataloader(validation_data)
        
        # set device for training operations based on user configuration        
        logger.info('Setting device for training operations based on user configuration')                 
        device = DeviceConfig(self.configuration) 
        device.set_device() 

        # initialize the model serializer and create checkpoint folder
        logger.info('Building FeXT AutoEncoder model based on user configuration') 
        modser = ModelSerializer() 
        checkpoint_path = modser.create_checkpoint_folder()
        # initialize and build FEXT Autoencoder
        autoencoder = FeXTAutoEncoder(self.configuration)           
        model = autoencoder.get_model(model_summary=True) 

        # check worker status to allow interruption
        check_thread_status(worker)   

        # generate training log report and graphviz plot for the model layout               
        modser.save_model_plot(model, checkpoint_path) 
        # perform training and save model at the end
        logger.info('Starting FeXT AutoEncoder training') 
        trainer = ModelTraining(self.configuration)  
        trainer.train_model(
            model, train_dataset, validation_dataset, checkpoint_path, 
            progress_callback=progress_callback, worker=worker)      
        
    #--------------------------------------------------------------------------
    def resume_training_pipeline(self, selected_checkpoint, progress_callback=None, worker=None):
        logger.info(f'Loading {selected_checkpoint} checkpoint from pretrained models') 
        modser = ModelSerializer()         
        model, train_config, session, checkpoint_path = modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  
        
        # set device for training operations based on user configuration
        logger.info('Setting device for training operations based on user configuration')                 
        device = DeviceConfig(self.configuration) 
        device.set_device() 

        logger.info('Preparing dataset of images based on splitting sizes')  
        sample_size = train_config.get("train_sample_size", 1.0)
        serializer = DataSerializer(self.database, train_config)  
        images_paths = serializer.get_images_path_from_directory(IMG_PATH, sample_size)
        splitter = TrainValidationSplit(train_config) 
        train_data, validation_data = splitter.split_train_and_validation(images_paths)     

        # create the tf.datasets using the previously initialized generators 
        logger.info('Building model data loaders with prefetching and parallel processing') 
        builder = ImageDataLoader(train_config)          
        train_dataset = builder.build_training_dataloader(train_data)
        validation_dataset = builder.build_training_dataloader(validation_data)

        # check worker status to allow interruption
        check_thread_status(worker)         
                            
        # resume training from pretrained model    
        logger.info(f'Resuming training from checkpoint {selected_checkpoint}') 
        trainer = ModelTraining(self.configuration) 
        trainer.resume_training(
            model, train_dataset, validation_dataset, checkpoint_path, session,
            progress_callback=progress_callback, worker=worker)
        
    #--------------------------------------------------------------------------
    def run_inference_pipeline(self, selected_checkpoint, progress_callback=None, worker=None):
        logger.info(f'Loading {selected_checkpoint} checkpoint from pretrained models')
        modser = ModelSerializer()         
        model, train_config, session, checkpoint_path = modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  

        # setting device for training         
        device = DeviceConfig(self.configuration) 
        device.set_device()

        # select images from the inference folder and retrieve current paths     
        serializer = DataSerializer(self.database, train_config)     
        images_paths = serializer.get_images_path_from_directory(INFERENCE_INPUT_PATH)
        logger.info(f'{len(images_paths)} images have been found as inference input')  

        # check worker status to allow interruption
        check_thread_status(worker)   
             
        # extract features from images using the encoder output, the image encoder
        # takes the list of images path from inference as input    
        encoder = ImageEncoding(model, train_config, checkpoint_path)  
        logger.info(f'Start encoding images using model {selected_checkpoint}')  
        encoder.encode_images_features(
            images_paths, progress_callback=progress_callback, worker=worker) 
        logger.info('Encoded images have been saved as .npy')
           
  
