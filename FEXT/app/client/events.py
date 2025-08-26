import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QImage, QPixmap

from FEXT.app.utils.data.serializer import DataSerializer, ModelSerializer
from FEXT.app.utils.data.loader import ImageDataLoader
from FEXT.app.utils.data.process import TrainValidationSplit
from FEXT.app.utils.learning.device import DeviceConfig
from FEXT.app.utils.learning.training.fitting import ModelTraining
from FEXT.app.utils.learning.models.autoencoder import FeXTAutoEncoders
from FEXT.app.utils.learning.inference.encoding import ImageEncoding
from FEXT.app.utils.validation.images import ImageAnalysis
from FEXT.app.utils.validation.checkpoints import ModelEvaluationSummary, ImageReconstruction
from FEXT.app.client.workers import check_thread_status

from FEXT.app.constants import IMG_PATH, INFERENCE_INPUT_PATH
from FEXT.app.logger import logger


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

    def __init__(self, configuration : dict): 
        self.serializer = DataSerializer() 
        self.modser = ModelSerializer()    
        self.inference_batch_size = configuration.get('inference_batch_size', 32)
        self.configuration = configuration  

    #--------------------------------------------------------------------------
    def load_img_path(self, path, sample_size=1.0):
        return self.serializer.get_img_path_from_directory(path, sample_size) 
        
    #--------------------------------------------------------------------------
    def run_dataset_evaluation_pipeline(self, metrics : list[str], progress_callback=None, worker=None):
        # get images path from the dataset folder and select a randomized fraction    
        sample_size = self.configuration.get("sample_size", 1.0)
        images_paths = self.serializer.get_img_path_from_directory(IMG_PATH, sample_size)
        logger.info(f'The image dataset is composed of {len(images_paths)} images')  
        analyzer = ImageAnalysis(self.configuration) 
        images = []

        # Mapping metric name to method and arguments
        metric_map = {
            'image_statistics': analyzer.calculate_image_statistics,
            'pixels_distribution': analyzer.calculate_pixel_intensity_distribution,
            'image_exposure': analyzer.calculate_exposure_metrics,
            'image_entropy': analyzer.calculate_entropy,
            'image_sharpness': analyzer.calculate_sharpness_metrics,
            'image_colorimetry': analyzer.calculate_color_metrics,
            'image_edges': analyzer.calculate_edge_metrics,
            'image_texture': analyzer.calculate_texture_lbp_metrics}

        for metric in metrics:
            if metric in metric_map:
                # check worker status to allow interruption
                check_thread_status(worker)
                metric_name = metric.replace('_', ' ').title()
                logger.info(f'Current metric: {metric_name}')  
                result = metric_map[metric](
                    images_paths, progress_callback=progress_callback, worker=worker)
                images.append(result)   

        return images
    
    #--------------------------------------------------------------------------
    def get_checkpoints_summary(self, progress_callback=None, worker=None): 
        summarizer = ModelEvaluationSummary(self.configuration)    
        checkpoints_summary = summarizer.get_checkpoints_summary(
            progress_callback=progress_callback, worker=worker) 
        logger.info(f'Checkpoints summary has been created for {checkpoints_summary.shape[0]} models')   
    
    #--------------------------------------------------------------------------
    def run_model_evaluation_pipeline(self, metrics, selected_checkpoint, progress_callback=None, worker=None):    
        if selected_checkpoint is None:
            logger.warning('No checkpoint selected for resuming training')
            return
           
        logger.info(f'Loading {selected_checkpoint} checkpoint')
        model, train_config, session, checkpoint_path = self.modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  

        # set device for training operations
        logger.info('Setting device for training operations')                
        device = DeviceConfig(self.configuration)
        device.set_device()  

        # isolate the encoder from the autoencoder model   
        encoder = ImageEncoding(model, train_config, checkpoint_path)
        encoder_model = encoder.encoder_model 

        logger.info('Preparing dataset of images based on splitting sizes')  
        sample_size = train_config.get("train_sample_size", 1.0)        
        images_paths = self.serializer.get_img_path_from_directory(IMG_PATH, sample_size)
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
            summarizer = ModelEvaluationSummary(self.configuration)       
            summarizer.get_evaluation_report(model, validation_dataset, worker=worker)              

        if 'image_reconstruction' in metrics:
            logger.info('Current metric: image reconstruction')
            validator = ImageReconstruction(self.configuration, model, checkpoint_path)      
            images.append(validator.visualize_reconstructed_images(
                validation_images, progress_callback=progress_callback, worker=worker))   
            logger.info('Image reconstruction analysis successfully performed')    

        return images   
         

###############################################################################
class ModelEvents:

    def __init__(self, configuration : dict):
        self.serializer = DataSerializer()  
        self.modser = ModelSerializer()
        self.configuration = configuration 

    #--------------------------------------------------------------------------
    def get_available_checkpoints(self):        
        return self.modser.scan_checkpoints_folder()
            
    #--------------------------------------------------------------------------
    def run_training_pipeline(self, progress_callback=None, worker=None):  
        logger.info('Preparing dataset of images based on splitting sizes')        
        sample_size = self.configuration.get("train_sample_size", 1.0)
        images_paths = self.serializer.get_img_path_from_directory(IMG_PATH, sample_size)
        splitter = TrainValidationSplit(self.configuration) 
        train_data, validation_data = splitter.split_train_and_validation(images_paths)
        
        # create the tf.datasets using the previously initialized generators 
        logger.info('Building model data loaders with prefetching and parallel processing')     
        builder = ImageDataLoader(self.configuration)          
        train_dataset = builder.build_training_dataloader(train_data)
        validation_dataset = builder.build_training_dataloader(validation_data)

        # check worker status to allow interruption
        check_thread_status(worker)
        
        # set device for training operations        
        logger.info('Setting device for training operations')                 
        device = DeviceConfig(self.configuration) 
        device.set_device() 

        # initialize the model serializer and create checkpoint folder
        model_name = self.configuration.get('selected_model', None)        
        checkpoint_path = self.modser.create_checkpoint_folder(model_name)

        # initialize and build FEXT Autoencoder
        logger.info(f'Building {model_name} model') 
        autoencoder = FeXTAutoEncoders(self.configuration)           
        model = autoencoder.get_selected_model(model_summary=True)
        # generate training log report and graphviz plot for the model layout               
        self.modser.save_model_plot(model, checkpoint_path)        
        
        # perform training and save model at the end
        logger.info('Starting FeXT AutoEncoder training') 
        trainer = ModelTraining(self.configuration)  
        model, history = trainer.train_model(
            model, train_dataset, validation_dataset, checkpoint_path, 
            progress_callback=progress_callback, worker=worker)  

        self.modser.save_pretrained_model(model, checkpoint_path)       
        self.modser.save_training_configuration(
            checkpoint_path, history, self.configuration)    
        
    #--------------------------------------------------------------------------
    def resume_training_pipeline(self, selected_checkpoint, progress_callback=None, worker=None):
        logger.info(f'Loading {selected_checkpoint} checkpoint') 
        model, train_config, session, checkpoint_path = self.modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  
        
        # set device for training operations
        logger.info('Setting device for training operations')                 
        device = DeviceConfig(self.configuration) 
        device.set_device() 

        logger.info('Preparing dataset of images based on splitting sizes')
        sample_size = train_config.get("train_sample_size", 1.0) 
        images_paths = self.serializer.get_img_path_from_directory(IMG_PATH, sample_size)
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
        additional_epochs = self.configuration.get('additional_epochs', 10)
        trainer = ModelTraining(train_config) 
        model, history = trainer.resume_training(
            model, train_dataset, validation_dataset, checkpoint_path, session,
            additional_epochs, progress_callback=progress_callback, worker=worker)
        
        self.modser.save_pretrained_model(model, checkpoint_path)       
        self.modser.save_training_configuration(
            checkpoint_path, history, self.configuration)
        
    #--------------------------------------------------------------------------
    def run_inference_pipeline(self, selected_checkpoint, progress_callback=None, worker=None):
        logger.info(f'Loading {selected_checkpoint} checkpoint')        
        model, train_config, _, checkpoint_path = self.modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  

        # setting device for training         
        device = DeviceConfig(self.configuration) 
        device.set_device()

        # select images from the inference folder and retrieve current paths 
        images_paths = self.serializer.get_img_path_from_directory(INFERENCE_INPUT_PATH)
        logger.info(f'{len(images_paths)} images have been found as inference input')  

        # check worker status to allow interruption
        check_thread_status(worker)   
             
        # extract features from images using the encoder output, the image encoder
        # takes the list of images path from inference as input    
        encoder = ImageEncoding(model, train_config, checkpoint_path)  
        logger.info(f'Start encoding images using model {selected_checkpoint}')  
        encoder.encode_img_features(
            images_paths, progress_callback=progress_callback, worker=worker) 
        logger.info('Encoded images have been saved as .npy')
           
  
