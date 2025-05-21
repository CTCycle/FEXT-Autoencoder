from functools import partial
from FEXT.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, Slot, QThreadPool, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QPushButton, QRadioButton, QCheckBox, QDoubleSpinBox, 
                               QSpinBox, QComboBox, QProgressBar, QGraphicsScene, 
                               QGraphicsPixmapItem, QGraphicsView)

from FEXT.commons.configuration import Configuration
from FEXT.commons.interface.events import ValidationEvents, TrainingEvents, InferenceEvents
from FEXT.commons.interface.workers import Worker
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
             
        self.metrics = []
        self.images_path = {'data' : [], 'inference' : []}
        self.image_pixmaps = []
        self.plot_pixmaps = []
        self.inference_pixmaps = []
        self.model_eval_pixmaps = []
        self.current_fig = 0
             
        # Internal state for checkpoints        
        self.selected_checkpoint = None
        
        # initial settings
        self.config_manager = Configuration()
        self.configuration = self.config_manager.get_configuration()
    
        self.threadpool = QThreadPool.globalInstance()
        self._validation_worker = None
        self._training_worker = None         

        # --- Create persistent handlers ---
        self.validation_handler = ValidationEvents(self.configuration)
        self.training_handler = TrainingEvents(self.configuration)
        self.inference_handler = InferenceEvents(self.configuration)

        # setup UI elements
        self._set_states()
        self.widgets = {}
        self._setup_configuration([
            # 1. dataset tab page
            (QCheckBox,'getStatsAnalysis','get_image_stats'),
            (QCheckBox,'getPixDist','get_pixels_dist'),
            (QPushButton,'getImgMetrics','get_img_metrics'),
            (QSpinBox,'seed','general_seed'),
            (QDoubleSpinBox,'sampleSize','sample_size'),
            (QPushButton,'loadImgDataset','load_source_images'),
            (QPushButton,'dataTabPreviousImg','data_tab_prev_img'),
            (QPushButton,'dataTabNextImg','data_tab_next_img'),
            (QPushButton,'dataTabClearImg','data_tab_clear_img'),
            (QRadioButton,'viewPlots','set_plot_view'),
            (QRadioButton,'viewImages','set_image_view'),  
            (QProgressBar,'dataProgressBar','data_progress_bar'),          
            # 2. training tab page    
            (QCheckBox,'imgAugment','img_augmentation'),
            (QCheckBox,'setShuffle','use_shuffle'),
            (QDoubleSpinBox,'trainSampleSize','train_sample_size'),            
            (QDoubleSpinBox,'validationSize','validation_size'),
            (QSpinBox,'shuffleSize','shuffle_size'),
            (QRadioButton,'setCPU','use_CPU'),
            (QRadioButton,'setGPU','use_GPU'),
            (QSpinBox,'deviceID','device_ID'),
            (QSpinBox,'numWorkers','num_workers'),
            (QCheckBox,'runTensorboard','use_tensorboard'),
            (QCheckBox,'realTimeHistory','get_real_time_history'),
            (QCheckBox,'saveCheckpoints','save_checkpoints'),
            (QSpinBox,'trainSeed','train_seed'),
            (QSpinBox,'splitSeed','split_seed'),
            (QSpinBox,'numEpochs','epochs'),
            (QSpinBox,'batchSize','batch_size'),            
            (QSpinBox,'saveCPFrequency','save_cp_frequency'),
            (QCheckBox,'useScheduler','LR_scheduler'), 
            (QDoubleSpinBox,'initialLearningRate','initial_LR'),
            (QDoubleSpinBox,'targetLearningRate','target_LR'),            
            (QSpinBox,'constantSteps','constant_steps'),
            (QSpinBox,'decaySteps','decay_steps'), 
            (QCheckBox,'mixedPrecision','use_mixed_precision'),
            (QCheckBox,'compileJIT','use_JIT_compiler'),   
            (QComboBox,'backendJIT','jit_backend'),         
            (QSpinBox,'initialNeurons','initial_neurons'),
            (QSpinBox,'bottleneckNeurons','bottleneck_neurons'),
            (QSpinBox,'numAdditionalEpochs','additional_epochs'),
            (QComboBox,'checkpointsList','checkpoints_list'),           
            (QPushButton,'refreshCheckpoints','refresh_checkpoints'),
            (QPushButton,'startTraining','start_training'),
            (QPushButton,'resumeTraining','resume_training'),            
            (QProgressBar,'trainingProgressBar','train_progress_bar'),
            # 3. model evaluation tab page
            (QPushButton,'evalTabPreviousImg','eval_tab_prev_img'),
            (QPushButton,'evalTabNextImg','eval_tab_next_img'),
            (QPushButton,'evalTabClearImg','eval_tab_clear_img'),    
            # 4. inference tab page        
            (QPushButton,'encodeImages','encode_images'),
            (QPushButton,'loadInferenceImages','load_inference_images'),
            (QPushButton,'inferTabPreviousImg','infer_tab_prev_img'),
            (QPushButton,'inferTabNextImg','infer_tab_next_img'),
            (QPushButton,'inferTabClearImg','infer_tab_clear_img'),
            ])
        
        self._connect_signals([
            # dataset tab
            ('get_image_stats','toggled',self._update_metrics),
            ('get_pixels_dist','toggled',self._update_metrics),
            ('get_img_metrics','clicked',self.compute_image_metrics),
            ('load_source_images','clicked', self.load_image_dataset),
            ('data_tab_prev_img', 'clicked', lambda: self.show_previous_figure("imageCanvas")),
            ('data_tab_next_img', 'clicked', lambda: self.show_next_figure("imageCanvas")),
            ('data_tab_clear_img', 'clicked', lambda: self.clear_figures("imageCanvas")),
            ('set_plot_view', 'toggled', lambda: self._update_graphics_view("imageCanvas")),
            ('set_image_view', 'toggled', lambda: self._update_graphics_view("imageCanvas")),
            # training tab
            ('checkpoints_list','currentTextChanged',self.select_checkpoint),
            ('refresh_checkpoints','clicked',self.load_checkpoints),
            ('start_training','clicked',self.train_from_scratch),
            ('resume_training','clicked',self.resume_training_from_checkpoint),
            # model evaluation tab
            ('eval_tab_prev_img','clicked', lambda: self.show_previous_figure("modelEvalCanvas")),            
            ('eval_tab_next_img','clicked', lambda: self.show_next_figure("modelEvalCanvas")),
            ('eval_tab_clear_img','clicked', lambda: self.clear_figures("modelEvalCanvas")),
            # inference tab
            ('encode_images','clicked',self.encode_images_with_checkpoint),
            ('load_inference_images','clicked', self.load_image_dataset),
            ('infer_tab_prev_img','clicked', lambda: self.show_previous_figure("inferenceImgCanvas")),
            ('infer_tab_next_img','clicked', lambda: self.show_next_figure("inferenceImgCanvas")),
            ('infer_tab_clear_img','clicked', lambda: self.clear_figures("inferenceImgCanvas")),
        ]) 
        
        self._auto_connect_settings() 
        self.use_GPU.toggled.connect(self._update_device)
        self.use_CPU.toggled.connect(self._update_device)
        
        # Initial population of dynamic UI elements
        self.load_checkpoints()
        self._set_graphics()           

    # ------------------- Helpers for Per-Setting Updates -------------------
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
            ('img_augmentation', 'toggled', 'img_augmentation'),
            ('use_shuffle', 'toggled', 'shuffle_dataset'),
            ('num_workers', 'valueChanged', 'num_workers'),
            ('use_mixed_precision', 'toggled', 'mixed_precision'),
            ('use_JIT_compiler', 'toggled', 'use_jit_compiler'),
            ('jit_backend', 'currentTextChanged', 'jit_backend'),
            ('use_tensorboard', 'toggled', 'run_tensorboard'),
            ('get_real_time_history', 'toggled', 'real_time_history'),
            ('save_checkpoints', 'toggled', 'save_checkpoints'),
            ('LR_scheduler', 'toggled', 'use_lr_scheduler'),
            ('general_seed', 'valueChanged', 'general_seed'),
            ('split_seed', 'valueChanged', 'split_seed'),
            ('train_seed', 'valueChanged', 'train_seed'),
            ('shuffle_size', 'valueChanged', 'shuffle_size'),
            ('epochs', 'valueChanged', 'epochs'),
            ('additional_epochs', 'valueChanged', 'additional_epochs'),
            ('initial_neurons', 'valueChanged', 'initial_neurons'),
            ('bottleneck_neurons', 'valueChanged', 'bottleneck_neurons'),
            ('batch_size', 'valueChanged', 'batch_size'),
            ('device_ID', 'valueChanged', 'device_id'),
            ('sample_size', 'valueChanged', 'sample_size'),
            ('train_sample_size', 'valueChanged', 'train_sample_size'),
            ('validation_size', 'valueChanged', 'validation_size'),
        ]
        for attr, signal_name, config_key in connections:
            widget = self.widgets[attr]
            self.connect_update_setting(widget, signal_name, config_key)

    #--------------------------------------------------------------------------
    def _update_device(self):
        device = 'GPU' if self.use_GPU.isChecked() else 'CPU'
        self.config_manager.update_value('device', device)

    # [SHOW WINDOW]
    ###########################################################################
    def show(self):        
        self.main_win.show()   

    # [HELPERS FOR SETTING CONNECTIONS]
    ###########################################################################
    def _set_states(self):         
        self.progress_bar = self.main_win.findChild(QProgressBar, "progressBar")        
        self.progress_bar.setValue(0) 

    #--------------------------------------------------------------------------
    def _set_graphics(self):
        self.graphics = {}
        CANVAS_NAMES = ["imageCanvas", "inferenceImgCanvas", "modelEvalCanvas"]
        for canvas_name in CANVAS_NAMES:
            view = self.main_win.findChild(QGraphicsView, canvas_name)
            scene = QGraphicsScene()
            pixmap_item = QGraphicsPixmapItem()
            pixmap_item.setTransformationMode(Qt.SmoothTransformation)
            scene.addItem(pixmap_item)
            view.setScene(scene)
            view.setRenderHint(QPainter.Antialiasing, True)
            view.setRenderHint(QPainter.SmoothPixmapTransform, True)
            view.setRenderHint(QPainter.TextAntialiasing, True)
            self.graphics[canvas_name] = {
                'view': view,
                'scene': scene,
                'pixmap_item': pixmap_item
            }

    #--------------------------------------------------------------------------
    def _connect_button(self, button_name: str, slot):        
        button = self.main_win.findChild(QPushButton, button_name)
        button.clicked.connect(slot) 

    #--------------------------------------------------------------------------
    def _connect_combo_box(self, combo_name: str, slot):        
        combo = self.main_win.findChild(QComboBox, combo_name)
        combo.currentTextChanged.connect(slot)

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

    #--------------------------------------------------------------------------
    def select_image_source(self, canvas_name: str):
        if canvas_name == "imageCanvas":
            source = self.plot_pixmaps if self.set_plot_view.isChecked() else self.image_pixmaps
        elif canvas_name == "inferenceImgCanvas":
            source = self.inference_pixmaps
        elif canvas_name == "modelEvalCanvas":
            source = self.model_eval_pixmaps 

        return source         
   
    # [SLOT]
    ###########################################################################
    @Slot()
    def _update_metrics(self):
        for name, box in [('image_stats', self.get_image_stats), 
                          ('pixels_distribution', self.get_pixels_dist)]:
            if box.isChecked():
                self.metrics.append(name) if name not in self.metrics else None                    
            else:
                self.metrics.remove(name) if name in self.metrics else None

    #--------------------------------------------------------------------------
    @Slot()
    def compute_image_metrics(self):  
        if not self.metrics:
            return None
        
        self.get_img_metrics.setEnabled(False)
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.configuration)       
        # send message to status bar
        self._send_message("Calculating image dataset evaluation metrics...") 
        # initialize worker for asynchronous loading of the dataset
        # functions that are passed to the worker will be executed in a separate thread
        self._validation_worker = Worker(
            self.validation_handler.run_dataset_evaluation_pipeline,
            self.metrics)                
        worker = self._validation_worker

        # inject the progress signal into the worker   
        self.progress_bar.setValue(0)    
        worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.finished.connect(self.on_metrics_calculated)
        worker.signals.error.connect(self.on_metrics_error)
        self.threadpool.start(worker)       

    #--------------------------------------------------------------------------
    @Slot()
    def train_from_scratch(self):  
        self.start_training.setEnabled(False)
        self.configuration = self.config_manager.get_configuration() 
        self.training_handler = TrainingEvents(self.configuration)         
  
        # send message to status bar
        self._send_message("Training FEXT Autoencoder model from scratch...") 
        # initialize worker for asynchronous loading of the dataset
        # functions that are passed to the worker will be executed in a separate thread
        self._training_worker = Worker(self.training_handler.run_training_pipeline)                            
        worker = self._training_worker

        # inject the progress signal into the worker   
        self.progress_bar.setValue(0)    
        worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.finished.connect(self.on_train_finished)
        worker.signals.error.connect(self.on_train_error)
        self.threadpool.start(worker)    

    #--------------------------------------------------------------------------
    @Slot()
    def resume_training_from_checkpoint(self):  
        self.resume_training.setEnabled(False)
        self.configuration = self.config_manager.get_configuration() 
        self.training_handler = TrainingEvents(self.configuration)   

        # send message to status bar
        self._send_message(f"Resume training from checkpoint {self.selected_checkpoint}") 
        # initialize worker for asynchronous loading of the dataset
        # functions that are passed to the worker will be executed in a separate thread
        self._training_worker = Worker(
            self.training_handler.resume_training_pipeline,
            self.selected_checkpoint)                            
        worker = self._training_worker

        # inject the progress signal into the worker   
        self.progress_bar.setValue(0)    
        worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.finished.connect(self.on_train_finished)
        worker.signals.error.connect(self.on_train_error)
        self.threadpool.start(worker)    
   
    #--------------------------------------------------------------------------
    @Slot()
    def load_checkpoints(self):       
        checkpoints = self.training_handler.get_available_checkpoints()
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
    def encode_images_with_checkpoint(self):  
        self.encode_images.setEnabled(False)
        self.configuration = self.config_manager.get_configuration() 
        self.training_handler = InferenceEvents(self.configuration)   

        # send message to status bar
        self._send_message(f"Encoding images with {self.selected_checkpoint}") 
        # initialize worker for asynchronous loading of the dataset
        # functions that are passed to the worker will be executed in a separate thread
        self._training_worker = Worker(
            self.training_handler.run_inference_pipeline,
            self.selected_checkpoint)                            
        worker = self._training_worker

        # inject the progress signal into the worker   
        self.progress_bar.setValue(0)    
        worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.finished.connect(self.on_inference_finished)
        worker.signals.error.connect(self.on_inference_error)
        self.threadpool.start(worker)           

    #--------------------------------------------------------------------------
    @Slot(str)
    def _update_graphics_view(self, canvas_name="imageCanvas"): 
        source = self.select_image_source(canvas_name)        
        if not source:
            return
        
        raw_pix = source[self.current_fig] if len(source) > 1 else source[0]       
        view = self.graphics[canvas_name]['view']
        pixmap_item = self.graphics[canvas_name]['pixmap_item']
        scene = self.graphics[canvas_name]['scene']
        view_size = view.viewport().size()
        scaled = raw_pix.scaled(
            view_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation)
        pixmap_item.setPixmap(scaled)
        scene.setSceneRect(scaled.rect())

    #--------------------------------------------------------------------------
    @Slot()
    def load_image_dataset(self):        
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.configuration)
        self.images_path['data'] = []        
        self.images_path['data'].extend(self.validation_handler.load_images_path()) 
        self.current_fig = 0
        self.image_pixmaps = []        
        self.image_pixmaps.append(
            self.validation_handler.load_image_as_pixmap(
                self.images_path['data'][self.current_fig])) 
        self._update_graphics_view("imageCanvas")
            

    #--------------------------------------------------------------------------
    @Slot(str)
    def show_previous_figure(self, canvas_name="imageCanvas"):        
        if self.current_fig > 0:            
            self.current_fig -= 1
            if self.set_image_view.isChecked():
                self.image_pixmaps = []
                self.image_pixmaps.append(
                self.validation_handler.load_image_as_pixmap(
                self.images_path['data'][self.current_fig])) 
            
            self._update_graphics_view(canvas_name)

    #--------------------------------------------------------------------------
    @Slot(str)
    def show_next_figure(self, canvas_name="imageCanvas"):
        source = self.select_image_source(canvas_name)
        if len(source) > 1 and self.current_fig < len(source):
            self.current_fig += 1
            self._update_graphics_view(canvas_name)
        
        elif self.set_image_view.isChecked() and len(source) == 1:
            self.current_fig += 1
            self.image_pixmaps = []
            self.image_pixmaps.append(
            self.validation_handler.load_image_as_pixmap(
            self.images_path['data'][self.current_fig]))              
            self._update_graphics_view(canvas_name)

    #--------------------------------------------------------------------------
    @Slot(str)
    def clear_figures(self, canvas_name="imageCanvas"):
        if canvas_name == "imageCanvas":            
            self.image_pixmaps = []
            self.plot_pixmaps = []
        elif canvas_name == "inferenceImgCanvas":
            self.inference_pixmaps = []
        elif canvas_name == "modelEvalCanvas":
            self.model_eval_pixmaps = []
            

    # [POSITIVE OUTCOME HANDLERS]
    ###########################################################################       
    def on_metrics_calculated(self, plots): 
        self.plot_pixmaps = []
        self.plot_pixmaps.extend([self.validation_handler.convert_fig_to_qpixmap(p) 
                for p in plots if plots is not None])
            
        self.current_fig = 0
        self._update_graphics_view("imageCanvas")
        self.validation_handler.handle_success(self.main_win, 'Figures have been generated')
        self.get_img_metrics.setEnabled(True)

    #--------------------------------------------------------------------------
    def on_train_finished(self, session):          
        self.training_handler.handle_success(
            self.main_win, 'Training session is over. Model has been saved')
        self.start_training.setEnabled(True) 
        self.resume_training.setEnabled(True)

    #--------------------------------------------------------------------------
    def on_inference_finished(self, session):          
        self.training_handler.handle_success(
            self.main_win, 'Training session is over. Model has been saved')
        self.encode_images.setEnabled(True)         

    # [NEGATIVE OUTCOME HANDLERS]
    ########################################################################### #    
    @Slot(tuple)
    def on_metrics_error(self, err_tb):
        self.training_handler.handle_error(self.main_win, err_tb) 
        self.get_img_metrics.setEnabled(True)

    @Slot(tuple)
    #--------------------------------------------------------------------------
    def on_train_error(self, err_tb):
        self.training_handler.handle_error(self.main_win, err_tb) 
        self.start_training.setEnabled(True) 
        self.resume_training.setEnabled(True)  

    @Slot(tuple)
    #--------------------------------------------------------------------------
    def on_inference_error(self, err_tb):
        self.inference_handler.handle_error(self.main_win, err_tb) 
        self.encode_images.setEnabled(True) 
         


        

    
       

    
