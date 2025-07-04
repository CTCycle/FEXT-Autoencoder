from FEXT.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

from functools import partial
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, Slot, QThreadPool, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (QPushButton, QRadioButton, QCheckBox, QDoubleSpinBox, 
                               QSpinBox, QComboBox, QProgressBar, QGraphicsScene, 
                               QGraphicsPixmapItem, QGraphicsView)

from FEXT.commons.utils.data.database import FEXTDatabase
from FEXT.commons.configuration import Configuration
from FEXT.commons.interface.events import GraphicsHandler, ValidationEvents, ModelEvents
from FEXT.commons.interface.workers import Worker
from FEXT.commons.constants import IMG_PATH, INFERENCE_INPUT_PATH
from FEXT.commons.logger import logger


###############################################################################
class MainWindow:
    
    def __init__(self, ui_file_path: str): 
        super().__init__()           
        loader = QUiLoader()
        ui_file = QFile(ui_file_path)
        ui_file.open(QIODevice.ReadOnly)
        self.main_win = loader.load(ui_file)
        ui_file.close()           

        # Checkpoint & metrics state
        self.selected_checkpoint = None
        self.selected_metrics = {'dataset': [], 'model': []}       
          
        # initial settings
        self.config_manager = Configuration()
        self.configuration = self.config_manager.get_configuration()
    
        # set thread pool for the workers
        self.threadpool = QThreadPool.globalInstance()
        self.worker = None
        self.worker_running = False        

        # initialize database
        self.database = FEXTDatabase(self.configuration)
        self.database.initialize_database()          

        # --- Create persistent handlers ---
        self.graphic_handler = GraphicsHandler()
        self.validation_handler = ValidationEvents(self.database, self.configuration)
        self.model_handler = ModelEvents(self.database, self.configuration)        

        # setup UI elements
        self._set_states()
        self.widgets = {}
        self._setup_configuration([ 
            # out of tab widgets
            (QPushButton,'refreshCheckpoints','refresh_checkpoints'),
            (QComboBox,'checkpointsList','checkpoints_list'),
            (QProgressBar,'progressBar','progress_bar'),      
            (QPushButton,'stopThread','stop_thread'),
            (QCheckBox,'deviceGPU','use_device_GPU'),    
            # 1. dataset tab page 
            # dataset evaluation group            
            (QSpinBox,'seed','general_seed'),
            (QDoubleSpinBox,'sampleSize','sample_size'),            
            (QCheckBox,'getPixDist','pixel_distribution_metric'),
            (QPushButton,'evaluateDataset','evaluate_dataset'),
            #  dataset processing group   
            # still nothing, to be implemented         
                      
            # 2. training tab page
            # dataset settings group    
            (QCheckBox,'imgAugment','img_augmentation'),
            (QCheckBox,'setShuffle','use_shuffle'),
            (QDoubleSpinBox,'trainSampleSize','train_sample_size'),
            (QDoubleSpinBox,'validationSize','validation_size'),
            (QSpinBox,'shuffleSize','shuffle_size'),
            # device settings group               
            (QSpinBox,'deviceID','device_ID'),
            (QSpinBox,'numWorkers','num_workers'),
            # training settings group
            (QCheckBox,'runTensorboard','use_tensorboard'),
            (QCheckBox,'realTimeHistory','real_time_history_callback'),
            (QCheckBox,'saveCheckpoints','save_checkpoints'),
            (QSpinBox,'saveCPFrequency','checkpoints_frequency'),            
            (QSpinBox,'numEpochs','epochs'),
            (QSpinBox,'batchSize','batch_size'),
            (QSpinBox,'trainSeed','train_seed'),
            (QSpinBox,'splitSeed','split_seed'), 
            # RL scheduler settings group
            (QCheckBox,'useScheduler','LR_scheduler'), 
            (QDoubleSpinBox,'initialLearningRate','initial_LR'),
            (QDoubleSpinBox,'targetLearningRate','target_LR'),            
            (QSpinBox,'constantSteps','constant_steps'),
            (QSpinBox,'decaySteps','decay_steps'), 
            # model settings group
            (QCheckBox,'mixedPrecision','use_mixed_precision'),
            (QCheckBox,'compileJIT','use_JIT_compiler'),   
            (QComboBox,'backendJIT','jit_backend'),         
            (QSpinBox,'initialNeurons','initial_neurons'),
            (QDoubleSpinBox,'dropoutRate','dropout_rate'), 
            # session settings group                   
            (QSpinBox,'numAdditionalEpochs','additional_epochs'),                      
            (QPushButton,'startTraining','start_training'),
            (QPushButton,'resumeTraining','resume_training'),            
            # 3. model evaluation tab page
            (QPushButton,'evaluateModel','model_evaluation'),
             
            (QPushButton,'checkpointSummary','checkpoints_summary'),
            (QCheckBox,'evalReport','get_evaluation_report'), 
            (QCheckBox,'imgReconstruction','image_reconstruction'), 
            (QSpinBox,'evalBatchSize','eval_batch_size'),     
            (QSpinBox,'numImages','num_evaluation_images'),           
            # 4. inference tab page              
            (QPushButton,'encodeImages','encode_images'),          
            # 5. Viewer tab
            (QPushButton,'loadImages','load_source_images'),
            (QPushButton,'previousImg','previous_image'),
            (QPushButton,'nextImg','next_image'),
            (QPushButton,'clearImg','clear_images'),
            (QRadioButton,'viewDataPlots','data_plots_view'),
            (QRadioButton,'viewEvalPlots','model_plots_view'),
            (QRadioButton,'viewInferenceImages','inference_images_view'),
            (QRadioButton,'viewTrainImages','train_images_view'),            
            ])
        
        self._connect_signals([  
            ('checkpoints_list','currentTextChanged',self.select_checkpoint), 
            ('refresh_checkpoints','clicked',self.load_checkpoints),
            ('stop_thread','clicked',self.stop_running_worker),          
            # 1. dataset tab page                      
            ('pixel_distribution_metric','toggled',self._update_metrics),
            ('evaluate_dataset','clicked',self.run_dataset_evaluation_pipeline),           
            # 2. training tab page               
            ('start_training','clicked',self.train_from_scratch),
            ('resume_training','clicked',self.resume_training_from_checkpoint),
            # 3. model evaluation tab page
            ('image_reconstruction','toggled',self._update_metrics),
            ('get_evaluation_report','toggled',self._update_metrics),            
            ('model_evaluation','clicked', self.run_model_evaluation_pipeline),
            ('checkpoints_summary','clicked',self.get_checkpoints_summary),                  
           
            # 4. inference tab page              
            ('encode_images','clicked',self.encode_images_with_checkpoint),            
            # 5. viewer tab page 
            ('data_plots_view', 'toggled', self._update_graphics_view),
            ('model_plots_view', 'toggled', self._update_graphics_view),
            ('inference_images_view', 'toggled', self._update_graphics_view), 
            ('train_images_view', 'toggled', self._update_graphics_view), 
            ('load_source_images','clicked', self.load_images),
            ('previous_image', 'clicked', self.show_previous_figure),
            ('next_image', 'clicked', self.show_next_figure),
            ('clear_images', 'clicked', self.clear_figures),            
            ]) 
        
        self._auto_connect_settings() 
               
        # Initial population of dynamic UI elements
        self.load_checkpoints()
        self._set_graphics() 


    # [SHOW WINDOW]
    ###########################################################################
    def show(self):        
        self.main_win.show()           

    # [HELPERS]
    ###########################################################################
    def connect_update_setting(self, widget, signal_name, config_key, getter=None):
        if getter is None:
            if isinstance(widget, (QCheckBox, QRadioButton)):
                getter = widget.isChecked
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                getter = widget.value
            elif isinstance(widget, QComboBox):
                getter = widget.currentText
           
        signal = getattr(widget, signal_name)
        signal.connect(partial(self._update_single_setting, config_key, getter))

    #--------------------------------------------------------------------------
    def _update_single_setting(self, config_key, getter, *args):
        value = getter()
        self.config_manager.update_value(config_key, value)

    #--------------------------------------------------------------------------
    def _auto_connect_settings(self):
        connections = [  
            ('use_device_GPU', 'toggled', 'use_device_GPU'),
            # 1. dataset tab page
            # dataset evaluation group
            ('general_seed', 'valueChanged', 'general_seed'),
            ('sample_size', 'valueChanged', 'sample_size'),
            #  dataset processing group   
            # still nothing, to be implemented

            # 2. training tab page
            # dataset settings group            
            ('img_augmentation', 'toggled', 'img_augmentation'),
            ('use_shuffle', 'toggled', 'shuffle_dataset'),
            ('shuffle_size', 'valueChanged', 'shuffle_size'),
            ('train_sample_size', 'valueChanged', 'train_sample_size'),            
            # device settings group
            ('device_ID', 'valueChanged', 'device_id'),
            ('num_workers', 'valueChanged', 'num_workers'),
            # training settings group
            ('use_tensorboard', 'toggled', 'run_tensorboard'),
            ('real_time_history_callback', 'toggled', 'real_time_history_callback'),
            ('save_checkpoints', 'toggled', 'save_checkpoints'),
            ('checkpoints_frequency', 'valueChanged', 'checkpoints_frequency'),
            ('epochs', 'valueChanged', 'epochs'),
            ('batch_size', 'valueChanged', 'batch_size'),
            ('train_seed', 'valueChanged', 'train_seed'),
            ('split_seed', 'valueChanged', 'split_seed'),            
            # RL scheduler settings group
            ('LR_scheduler', 'toggled', 'use_lr_scheduler'),
            ('initial_LR', 'valueChanged', 'initial_LR'),
            ('target_LR', 'valueChanged', 'target_LR'),
            ('constant_steps', 'valueChanged', 'constant_steps'),
            ('decay_steps', 'valueChanged', 'decay_steps'),
            # model settings group
            ('use_mixed_precision', 'toggled', 'mixed_precision'),
            ('use_JIT_compiler', 'toggled', 'use_jit_compiler'),
            ('jit_backend', 'currentTextChanged', 'jit_backend'),
            ('initial_neurons', 'valueChanged', 'initial_neurons'),
            ('dropout_rate', 'valueChanged', 'dropout_rate'),
            # session settings group
            ('additional_epochs', 'valueChanged', 'additional_epochs'),

            # 3. model evaluation tab page            
            ('eval_batch_size', 'valueChanged', 'eval_batch_size'),
            ('num_evaluation_images', 'valueChanged', 'num_evaluation_images'),
        
            # 4. inference tab page                     
            ('validation_size', 'valueChanged', 'validation_size')
            ]

        self.data_metrics = [
            ('pixels_distribution', self.pixel_distribution_metric)]
        self.model_metrics = [
            ('evaluation_report', self.get_evaluation_report),
            ('image_reconstruction', self.image_reconstruction)]                

        for attr, signal_name, config_key in connections:
            widget = self.widgets[attr]
            self.connect_update_setting(widget, signal_name, config_key)

    #--------------------------------------------------------------------------
    def _set_states(self):         
        self.progress_bar = self.main_win.findChild(QProgressBar, "progressBar")        
        self.progress_bar.setValue(0)

    #--------------------------------------------------------------------------
    def get_current_pixmaps_and_key(self):
        for radio, (pixmap_key, idx_key) in self.pixmap_source_map.items():
            if radio.isChecked():
                return self.pixmaps[pixmap_key], idx_key
        return [], None 

    #--------------------------------------------------------------------------
    def _set_graphics(self):
        self.graphics = {}        
        view = self.main_win.findChild(QGraphicsView, 'canvas')
        scene = QGraphicsScene()
        pixmap_item = QGraphicsPixmapItem()
        pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        scene.addItem(pixmap_item)
        view.setScene(scene)
        view.setRenderHint(QPainter.Antialiasing, True)
        view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        view.setRenderHint(QPainter.TextAntialiasing, True)
        self.graphics = {'view': view,
                         'scene': scene,
                         'pixmap_item': pixmap_item}        
                        
        self.pixmaps = {
            'train_images': [],         
            'inference_images': [],      
            'dataset_eval_images': [],  
            'model_eval_images': []}
        
        self.img_paths = {'train_images' : IMG_PATH,
                          'inference_images' : INFERENCE_INPUT_PATH}
            
        self.current_fig = {'train_images' : 0, 'inference_images' : 0,
                            'dataset_eval_images' : 0, 'model_eval_images' : 0}   

        self.pixmap_source_map = {
            self.data_plots_view: ("dataset_eval_images", "dataset_eval_images"),
            self.model_plots_view: ("model_eval_images", "model_eval_images"),
            self.inference_images_view: ("inference_images", "inference_images"),
            self.train_images_view: ("train_images", "train_images")}             

    #--------------------------------------------------------------------------
    def _connect_button(self, button_name: str, slot):        
        button = self.main_win.findChild(QPushButton, button_name)
        button.clicked.connect(slot) 

    #--------------------------------------------------------------------------
    def _connect_combo_box(self, combo_name: str, slot):        
        combo = self.main_win.findChild(QComboBox, combo_name)
        combo.currentTextChanged.connect(slot)

    #--------------------------------------------------------------------------
    def _start_worker(self, worker : Worker, on_finished, on_error, on_interrupted,
                      update_progress=True):
        if update_progress:       
            self.progress_bar.setValue(0)
            worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(on_error)        
        worker.signals.interrupted.connect(on_interrupted)
        self.threadpool.start(worker)
        self.worker_running = True

    #--------------------------------------------------------------------------
    def _send_message(self, message): 
        self.main_win.statusBar().showMessage(message)    

    # [SETUP]
    ###########################################################################
    def _setup_configuration(self, widget_defs):
        for cls, name, attr in widget_defs:
            w = self.main_win.findChild(cls, name)
            setattr(self, attr, w)
            self.widgets[attr] = w

    #--------------------------------------------------------------------------
    def _connect_signals(self, connections):
        for attr, signal, slot in connections:
            widget = self.widgets[attr]
            getattr(widget, signal).connect(slot)   
   
    # [SLOT]
    ###########################################################################
    # It's good practice to define methods that act as slots within the class
    # that manages the UI elements. These slots can then call methods on the
    # handler objects. Using @Slot decorator is optional but good practice
    #--------------------------------------------------------------------------
    Slot()
    def stop_running_worker(self):
        if self.worker is not None:
            self.worker.stop()       
        self._send_message("Interrupt requested. Waiting for threads to stop...")

    #--------------------------------------------------------------------------
    @Slot()
    def load_checkpoints(self):       
        checkpoints = self.model_handler.get_available_checkpoints()
        self.checkpoints_list.clear()
        if checkpoints:
            self.checkpoints_list.addItems(checkpoints)
            self.selected_checkpoint = checkpoints[0]
            self.checkpoints_list.setCurrentText(checkpoints[0])
        else:
            self.selected_checkpoint = None
            logger.warning("No checkpoints available")

    #--------------------------------------------------------------------------
    @Slot(str)
    def select_checkpoint(self, name: str):
        self.selected_checkpoint = name if name else None 

    #--------------------------------------------------------------------------
    @Slot()
    def _update_metrics(self):             
        self.selected_metrics['dataset'] = [
            name for name, box in self.data_metrics if box.isChecked()]
        self.selected_metrics['model'] = [
            name for name, box in self.model_metrics if box.isChecked()]
        
    #--------------------------------------------------------------------------
    # [GRAPHICS]
    #--------------------------------------------------------------------------
    @Slot(str)
    def _update_graphics_view(self):  
        pixmaps, idx_key = self.get_current_pixmaps_and_key()
        if not pixmaps or idx_key is None:
            self.graphics['pixmap_item'].setPixmap(QPixmap())
            self.graphics['scene'].setSceneRect(0, 0, 0, 0)
            return

        idx = self.current_fig.get(idx_key, 0)
        idx = min(idx, len(pixmaps) - 1)
        raw = pixmaps[idx]
        
        qpixmap = QPixmap(raw) if isinstance(raw, str) else raw
        view = self.graphics['view']
        pixmap_item = self.graphics['pixmap_item']
        scene = self.graphics['scene']
        view_size = view.viewport().size()
        scaled = qpixmap.scaled(
            view_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap_item.setPixmap(scaled)
        scene.setSceneRect(scaled.rect())     

    #--------------------------------------------------------------------------
    @Slot(str)
    def show_previous_figure(self):             
        pixmaps, idx_key = self.get_current_pixmaps_and_key()
        if not pixmaps or idx_key is None:
            return
        if self.current_fig[idx_key] > 0:
            self.current_fig[idx_key] -= 1
            self._update_graphics_view()

    #--------------------------------------------------------------------------
    @Slot(str)
    def show_next_figure(self):
        pixmaps, idx_key = self.get_current_pixmaps_and_key()
        if not pixmaps or idx_key is None:
            return
        if self.current_fig[idx_key] < len(pixmaps) - 1:
            self.current_fig[idx_key] += 1
            self._update_graphics_view()

    #--------------------------------------------------------------------------
    @Slot(str)
    def clear_figures(self):
        pixmaps, idx_key = self.get_current_pixmaps_and_key()
        if not pixmaps or idx_key is None:
            return
        self.pixmaps[idx_key].clear()
        self.current_fig[idx_key] = 0
        self._update_graphics_view()
        self.graphics['pixmap_item'].setPixmap(QPixmap())
        self.graphics['scene'].setSceneRect(0, 0, 0, 0)
        self.graphics['view'].viewport().update()

    #--------------------------------------------------------------------------    
    @Slot()
    def load_images(self):          
        pixmaps, idx_key = self.get_current_pixmaps_and_key()
        if idx_key not in self.img_paths.keys():
            return
        
        self.pixmaps[idx_key].clear()
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.database, self.configuration)
        
        img_paths = self.validation_handler.load_images_path(self.img_paths[idx_key])
        self.pixmaps[idx_key].extend(img_paths)
        self.current_fig[idx_key] = 0 
        self._update_graphics_view()    

    #--------------------------------------------------------------------------
    # [DATASET TAB]
    #--------------------------------------------------------------------------        
    @Slot()
    def run_dataset_evaluation_pipeline(self):  
        if not self.data_metrics:
            return 
        
        if self.worker_running:            
            return         
        
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.database, self.configuration)       
        # send message to status bar
        self._send_message("Calculating image dataset evaluation metrics...") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.validation_handler.run_dataset_evaluation_pipeline,
            self.selected_metrics['dataset'])   

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_dataset_evaluation_finished,
            on_error=self.on_evaluation_error,
            on_interrupted=self.on_task_interrupted)       

    #--------------------------------------------------------------------------
    # [TRAINING TAB]
    #-------------------------------------------------------------------------- 
    @Slot()
    def train_from_scratch(self):
        if self.worker_running:            
            return 
                  
        self.configuration = self.config_manager.get_configuration() 
        self.model_handler = ModelEvents(self.database, self.configuration)         
  
        # send message to status bar
        self._send_message("Training FEXT Autoencoder using a new model instance...")        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(self.model_handler.run_training_pipeline)                            
       
        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_train_finished,
            on_error=self.on_model_error,
            on_interrupted=self.on_task_interrupted)  

    #--------------------------------------------------------------------------
    @Slot()
    def resume_training_from_checkpoint(self): 
        if self.worker_running or not self.selected_checkpoint:            
            return         
              
        self.configuration = self.config_manager.get_configuration() 
        self.model_handler = ModelEvents(self.database, self.configuration)   

        # send message to status bar
        self._send_message(f"Resume training from checkpoint {self.selected_checkpoint}")         
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.model_handler.resume_training_pipeline,            
            self.selected_checkpoint)   

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_train_finished,
            on_error=self.on_model_error,
            on_interrupted=self.on_task_interrupted)

    #--------------------------------------------------------------------------
    # [MODEL EVALUATION TAB]
    #-------------------------------------------------------------------------- 
    @Slot()
    def run_model_evaluation_pipeline(self):  
        if self.worker_running:            
            return 

        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.database, self.configuration)         
        # send message to status bar
        self._send_message(f"Evaluating {self.select_checkpoint} performances... ")

        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.validation_handler.run_model_evaluation_pipeline,
            self.selected_metrics['model'], 
            self.selected_checkpoint)                
        
        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_model_evaluation_finished,
            on_error=self.on_model_error,
            on_interrupted=self.on_task_interrupted)     

    #-------------------------------------------------------------------------- 
    @Slot()
    def get_checkpoints_summary(self):       
        if self.worker_running:            
            return 
        
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.database, self.configuration)           
        # send message to status bar
        self._send_message("Generating checkpoints summary...") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(self.validation_handler.get_checkpoints_summary) 

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_model_evaluation_finished,
            on_error=self.on_model_error,
            on_interrupted=self.on_task_interrupted)  

    #--------------------------------------------------------------------------
    # [INFERENCE TAB]
    #--------------------------------------------------------------------------   
    @Slot()    
    def encode_images_with_checkpoint(self):  
        if self.worker_running:            
            return 
        
        self.configuration = self.config_manager.get_configuration() 
        self.model_handler = ModelEvents(self.database, self.configuration)            
        # send message to status bar
        self._send_message(f"Encoding images with {self.selected_checkpoint}") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.model_handler.run_inference_pipeline,
            self.selected_checkpoint)

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_inference_finished,
            on_error=self.on_model_error,
            on_interrupted=self.on_task_interrupted)


    ###########################################################################
    # [POSITIVE OUTCOME HANDLERS]
    ###########################################################################       
    def on_dataset_evaluation_finished(self, plots):   
        key = 'dataset_eval_images'      
        if plots:            
            self.pixmaps[key].extend(
                [self.graphic_handler.convert_fig_to_qpixmap(p) 
                 for p in plots])
            
        self.current_fig[key] = 0
        self._update_graphics_view()
        self.validation_handler.handle_success(self.main_win, 'Figures have been generated')
        self.worker_running = False

    #--------------------------------------------------------------------------
    def on_train_finished(self, session):          
        self.model_handler.handle_success(
            self.main_win, 'Training session is over. Model has been saved')
        self.worker_running = False

    #--------------------------------------------------------------------------
    def on_model_evaluation_finished(self, plots):  
        key = 'model_eval_images'         
        if plots is not None:            
            self.pixmaps[key].extend(
                [self.graphic_handler.convert_fig_to_qpixmap(p)
                for p in plots])
            
        self.current_fig[key] = 0
        self._update_graphics_view()
        self.validation_handler.handle_success(
            self.main_win, f'Model {self.selected_checkpoint} has been evaluated')
        self.worker_running = False

    #--------------------------------------------------------------------------
    def on_inference_finished(self, session):          
        self.model_handler.handle_success(
            self.main_win, 'Inference call has been terminated')
        self.worker_running = False


    ###########################################################################   
    # [NEGATIVE OUTCOME HANDLERS]
    ###########################################################################    
    @Slot(tuple)
    def on_evaluation_error(self, err_tb):
        self.validation_handler.handle_error(self.main_win, err_tb) 
        self.worker_running = False   

    #--------------------------------------------------------------------------
    @Slot() 
    def on_model_error(self, err_tb):
        self.model_handler.handle_error(self.main_win, err_tb) 
        self.worker_running = False  

    #--------------------------------------------------------------------------
    def on_task_interrupted(self):         
        self.progress_bar.setValue(0)
        self._send_message('Current task has been interrupted by user') 
        logger.warning('Current task has been interrupted by user')
        self.worker_running = False        
        
          
         


        

    
       

    
