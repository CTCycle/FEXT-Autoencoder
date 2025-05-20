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
       
        self.plots = []
        self.images = []
        self.metrics = []
        self.image_pixmaps = None
        self.plot_pixmaps = None
        self.inference_pixmaps = None
        self.model_eval_pixmaps = None
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
        # These objects will live as long as the MainWindow instance lives
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
            (QPushButton,'previousImg','prev_img'),
            (QPushButton,'nextImg','next_img'),
            (QPushButton,'clearImg','clear_img'),
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
            (QComboBox,'backendJIT','backend_jit'),         
            (QSpinBox,'initialNeurons','initial_neurons'),
            (QSpinBox,'bottleneckNeurons','bottleneck_neurons'),
            (QSpinBox,'numAdditionalEpochs','additional_epochs'),
            (QComboBox,'checkpointsList','checkpoints_list'),           
            (QPushButton,'refreshCheckpoints','refresh_checkpoints'),
            (QPushButton,'startTraining','start_training'),
            (QPushButton,'resumeTraining','resume_training'),            
            (QProgressBar,'trainingProgressBar','train_progress_bar'),
            
            
            # 4. inference tab page        
            (QPushButton,'encodeImages','encode_images'),
            ])
        
        self._connect_signals([
            # 1. dataset tab page
            ('get_image_stats','toggled',self._update_metrics),
            ('get_pixels_dist','toggled',self._update_metrics),
            ('get_img_metrics','clicked',self.compute_image_metrics),
            ('general_seed','valueChanged',self._update_settings),
            ('sample_size','valueChanged',self._update_settings),
            ('prev_img', 'clicked', lambda: self.show_previous_figure("imageCanvas")),
            ('next_img', 'clicked', lambda: self.show_next_figure("imageCanvas")),
            ('clear_img', 'clicked', lambda: self.clear_figures("imageCanvas")),
            ('set_plot_view', 'toggled', lambda: self._update_graphics_view("imageCanvas")),
            ('set_image_view', 'toggled', lambda: self._update_graphics_view("imageCanvas")),
            # 2. training tab page
            ('img_augmentation','toggled',self._update_settings),
            ('use_shuffle','toggled',self._update_settings),
            ('train_sample_size','valueChanged',self._update_settings),
            ('validation_size','valueChanged',self._update_settings),
            ('shuffle_size','valueChanged',self._update_settings),
            ('use_CPU','toggled',self._update_settings),
            ('use_GPU','toggled',self._update_settings),
            ('device_ID','valueChanged',self._update_settings),
            ('num_workers','valueChanged',self._update_settings),
            ('use_tensorboard','toggled',self._update_settings),
            ('get_real_time_history','toggled',self._update_settings),
            ('save_checkpoints','toggled',self._update_settings),
            ('train_seed','valueChanged',self._update_settings),
            ('split_seed','valueChanged',self._update_settings),
            ('epochs','valueChanged',self._update_settings),
            ('batch_size','valueChanged',self._update_settings),
            ('save_cp_frequency','valueChanged',self._update_settings),
            ('LR_scheduler','toggled',self._update_settings),
            ('initial_LR','valueChanged',self._update_settings),
            ('target_LR','valueChanged',self._update_settings),
            ('constant_steps','valueChanged',self._update_settings),
            ('decay_steps','valueChanged',self._update_settings),
            ('use_mixed_precision','toggled',self._update_settings),
            ('use_JIT_compiler','toggled',self._update_settings),
            ('backend_jit','currentTextChanged',self._update_settings),
            ('initial_neurons','valueChanged',self._update_settings),
            ('bottleneck_neurons','valueChanged',self._update_settings),
            ('additional_epochs','valueChanged',self._update_settings),
            ('checkpoints_list','currentTextChanged',self.select_checkpoint),
            ('refresh_checkpoints','clicked',self.load_checkpoints),
            ('start_training','clicked',self.train_from_scratch),
            ('resume_training','clicked',self.resume_training_from_checkpoint),

            # 4. inference tab page
            ('encode_images','clicked',self.encode_images_with_checkpoint),
            ]) 
            
        # Initial population of dynamic UI elements
        self.load_checkpoints()
        self._set_graphics()       

        

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
       
    # [SLOT]
    ###########################################################################
    # It's good practice to define methods that act as slots within the class
    # that manages the UI elements. These slots can then call methods on the
    # handler objects. Using @Slot decorator is optional but good practice
    #--------------------------------------------------------------------------
    @Slot()
    def _update_settings(self):        
        self.config_manager.update_value('img_augmentation', self.img_augmentation.isChecked())
        self.config_manager.update_value('shuffle_dataset', self.use_shuffle.isChecked())
        self.config_manager.update_value('num_workers', self.num_workers.value())
        self.config_manager.update_value('mixed_precision', self.use_mixed_precision.isChecked())
        self.config_manager.update_value('use_jit_compiler', self.use_JIT_compiler.isChecked())        
        self.config_manager.update_value('jit_backend', self.backend_jit.currentText())
        self.config_manager.update_value('run_tensorboard', self.use_tensorboard.isChecked())
        self.config_manager.update_value('real_time_history', self.get_real_time_history.isChecked())
        self.config_manager.update_value('save_checkpoints', self.save_checkpoints.isChecked())
        self.config_manager.update_value('use_lr_scheduler', self.LR_scheduler.isChecked())       
        self.config_manager.update_value('general_seed', self.general_seed.value())
        self.config_manager.update_value('split_seed', self.split_seed.value())
        self.config_manager.update_value('train_seed', self.train_seed.value())
        self.config_manager.update_value('shuffle_size', self.shuffle_size.value())
        self.config_manager.update_value('epochs', self.epochs.value())
        self.config_manager.update_value('additional_epochs', self.additional_epochs.value())
        self.config_manager.update_value('initial_neurons', self.initial_neurons.value())
        self.config_manager.update_value('bottleneck_neurons', self.bottleneck_neurons.value())
        self.config_manager.update_value('batch_size', self.batch_size.value())
        self.config_manager.update_value('device_id', self.device_ID.value())
        self.config_manager.update_value('sample_size', self.sample_size.value())
        self.config_manager.update_value('train_sample_size', self.train_sample_size.value())
        self.config_manager.update_value('validation_size', self.validation_size.value())       

        self.device = 'GPU' if self.use_GPU.isChecked() else 'CPU'
        self.config_manager.update_value('device', self.device)

    #--------------------------------------------------------------------------
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
        if canvas_name == "imageCanvas":
            source = self.plot_pixmaps if self.set_plot_view.isChecked() else self.image_pixmaps
        elif canvas_name == "inferenceImgCanvas":
            source = self.inference_pixmaps
        elif canvas_name == "modelEvalCanvas":
            source = self.model_eval_pixmaps
        else:
            return
        if source:
            raw_pix = source[self.current_fig]
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
    @Slot(str)
    def show_previous_figure(self, canvas_name="imageCanvas"):
        if self.current_fig > 0:
            self.current_fig -= 1
            self._update_graphics_view(canvas_name)

    #--------------------------------------------------------------------------
    @Slot(str)
    def show_next_figure(self, canvas_name="imageCanvas"):
        if canvas_name == "imageCanvas":
            elements = len(self.plot_pixmaps) if self.set_plot_view.isChecked() \
                       else len(self.image_pixmaps)
        elif canvas_name == "inferenceImgCanvas":
            elements = len(self.inference_pixmaps)
        elif canvas_name == "modelEvalCanvas":
            elements = len(self.model_eval_pixmaps)       

        if self.current_fig < elements - 1:
            self.current_fig += 1
            self._update_graphics_view(canvas_name)

    #--------------------------------------------------------------------------
    @Slot(str)
    def clear_figures(self, canvas_name="imageCanvas"):
        if canvas_name == "imageCanvas":
            self.images = []
            self.image_pixmaps = None
        elif canvas_name == "inferenceImgCanvas":
            self.inference_pixmaps = None
        elif canvas_name == "modelEvalCanvas":
            self.model_eval_pixmaps = None
            

    # [POSITIVE OUTCOME HANDLERS]
    ###########################################################################       
    def on_metrics_calculated(self, plots):   
        self.plots.extend(plots) if plots else None
        self.plot_pixmaps = [
            self.validation_handler.convert_fig_to_qpixmap(p) for p in self.plots]
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
         


        

    
       

    
