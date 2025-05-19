from FEXT.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, Slot, QThreadPool, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QPushButton, QRadioButton, QCheckBox, QDoubleSpinBox, 
                               QSpinBox, QComboBox, QProgressBar, QGraphicsScene, 
                               QGraphicsPixmapItem, QGraphicsView)

from FEXT.commons.configuration import Configuration
from FEXT.commons.interface.events import ValidationEvents, TrainingEvents
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
       
        self.text_dataset = None
        self.tokenizers = None        
        self.plots = []       
        self.images = []
        self.metrics = []
        self.image_pixmaps = None
        self.plot_pixmaps = None
        self.current_fig = 0

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

        # setup UI elements
        self._set_states()
        self.widgets = {}
        self._setup_configuration([
            (QCheckBox,'getStatsAnalysis','get_image_stats'),
            (QCheckBox,'getPixDist','get_pixels_dist'),            
            (QCheckBox,'imgAugment','img_augmentation'),
            (QCheckBox,'setShuffle','use_shuffle'),
            (QCheckBox,'mixedPrecision','use_mixed_precision'),
            (QCheckBox,'compileJIT','use_JIT_compiler'),
            (QCheckBox,'runTensorboard','use_tensorboard'),
            (QCheckBox,'realTimeHistory','get_real_time_history'),
            (QCheckBox,'saveCheckpoints','save_checkpoints'),
            (QCheckBox,'useScheduler','LR_scheduler'), 
            (QSpinBox,'seed','general_seed'),
            (QSpinBox,'splitSeed','split_seed'),
            (QSpinBox,'trainSeed','train_seed'),
            (QSpinBox,'shuffleSize','shuffle_size'),
            (QSpinBox,'numEpochs','epochs'),
            (QSpinBox,'batchSize','batch_size'),
            (QSpinBox,'deviceID','device_ID'),
            (QSpinBox,'saveCPFrequency','save_cp_frequency'),
            (QSpinBox,'numWorkers','num_workers'),
            (QSpinBox,'constantSteps','constant_steps'),
            (QSpinBox,'decaySteps','decay_steps'),  
            (QDoubleSpinBox,'sampleSize','sample_size'),                    
            (QDoubleSpinBox,'trainSampleSize','train_sample_size'),
            (QDoubleSpinBox,'validationSize','validation_size'),
            (QDoubleSpinBox,'initialLearningRate','initial_LR'),
            (QDoubleSpinBox,'targetLearningRate','target_LR'),
            (QRadioButton,'setCPU','use_CPU'),
            (QRadioButton,'setGPU','use_GPU'),
            (QRadioButton,'viewPlots','set_plot_view'),
            (QRadioButton,'viewImages','set_image_view'),            
            (QComboBox,'backendJIT','backend_jit'),
            (QPushButton,'getImgMetrics','get_img_metrics'),
            (QPushButton,'previousImg','prev_img'),
            (QPushButton,'nextImg','next_img'),
            (QPushButton,'clearImg','clear_img'),
            (QPushButton,'startTraining','start_training'),
            (QPushButton,'resumeTraining','resume_training'),
            (QProgressBar,'dataProgressBar','data_progress_bar'),
            (QProgressBar,'trainingProgressBar','train_progress_bar')])
        
        self._connect_signals([
            ('img_augmentation','toggled',self._update_settings),
            ('use_shuffle','toggled',self._update_settings),                           
            ('use_mixed_precision','toggled',self._update_settings),
            ('use_JIT_compiler','toggled',self._update_settings),
            ('use_tensorboard','toggled',self._update_settings),
            ('get_real_time_history','toggled',self._update_settings),
            ('save_checkpoints','toggled',self._update_settings),
            ('LR_scheduler','toggled',self._update_settings),
            ('get_image_stats','toggled',self._update_settings),
            ('get_pixels_dist','toggled',self._update_settings),
            ('use_CPU','toggled',self._update_settings),
            ('use_GPU','toggled',self._update_settings),
            ('general_seed','valueChanged',self._update_settings),
            ('split_seed','valueChanged',self._update_settings),
            ('train_seed','valueChanged',self._update_settings),
            ('shuffle_size','valueChanged',self._update_settings),            
            ('epochs','valueChanged',self._update_settings),
            ('batch_size','valueChanged',self._update_settings),
            ('device_ID','valueChanged',self._update_settings),
            ('save_cp_frequency','valueChanged',self._update_settings),
            ('num_workers','valueChanged',self._update_settings),
            ('constant_steps','valueChanged',self._update_settings),
            ('decay_steps','valueChanged',self._update_settings),
            ('sample_size','valueChanged',self._update_settings),
            ('train_sample_size','valueChanged',self._update_settings),
            ('validation_size','valueChanged',self._update_settings),
            ('initial_LR','valueChanged',self._update_settings),
            ('target_LR','valueChanged',self._update_settings),
            ('set_plot_view','toggled',self._update_graphics_view),
            ('set_image_view','toggled',self._update_graphics_view),
            ('backend_jit','currentTextChanged',self.update_JIT_backend),
            ('get_img_metrics','clicked',self.compute_image_metrics),
            ('prev_img','clicked',self.show_previous_figure),
            ('next_img','clicked',self.show_next_figure),
            ('clear_img','clicked',self.clear_figures),
            ('start_training','clicked',self.train_from_scratch),
            ('resume_training','clicked',self.resume_training_from_checkpoint)]) 

        # --- prepare graphics view for figures ---
        self.view = self.main_win.findChild(QGraphicsView, "imageCanvas")
        self.scene = QGraphicsScene()
        self.pixmap_item = QGraphicsPixmapItem()
        # make pixmap scaling use smooth interpolation
        self.pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self.scene.addItem(self.pixmap_item)
        self.view.setScene(self.scene)
        # set canvas hints
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.view.setRenderHint(QPainter.TextAntialiasing, True)      


    # [SHOW WINDOW]
    ###########################################################################
    def show(self):        
        self.main_win.show()    

    # [HELPERS FOR SETTING CONNECTIONS]
    ###########################################################################
    def _set_states(self): 
        self.data_progress_bar = self.main_win.findChild(QProgressBar, "dataProgressBar")
        self.train_progress_bar = self.main_win.findChild(QProgressBar, "trainingProgressBar")
        self.data_progress_bar.setValue(0)  
        self.train_progress_bar.setValue(0)   

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
        self.config_manager.update_value('run_tensorboard', self.use_tensorboard.isChecked())
        self.config_manager.update_value('real_time_history', self.get_real_time_history.isChecked())
        self.config_manager.update_value('save_checkpoints', self.save_checkpoints.isChecked())
        self.config_manager.update_value('use_lr_scheduler', self.LR_scheduler.isChecked())       
        self.config_manager.update_value('general_seed', self.general_seed.value())
        self.config_manager.update_value('split_seed', self.split_seed.value())
        self.config_manager.update_value('train_seed', self.train_seed.value())
        self.config_manager.update_value('shuffle_size', self.shuffle_size.value())
        self.config_manager.update_value('epochs', self.epochs.value())
        self.config_manager.update_value('batch_size', self.batch_size.value())
        self.config_manager.update_value('device_id', self.device_ID.value())
        self.config_manager.update_value('sample_size', self.sample_size.value())
        self.config_manager.update_value('train_sample_size', self.train_sample_size.value())
        self.config_manager.update_value('validation_size', self.validation_size.value())

        self.device = 'GPU' if self.use_GPU.isChecked() else 'CPU'
        self.config_manager.update_value('device', self.device)        

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
        
        self.main_win.findChild(QPushButton, "getImgMetrics").setEnabled(False)
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
        self.data_progress_bar.setValue(0)    
        worker.signals.progress.connect(self.data_progress_bar.setValue)
        worker.signals.finished.connect(self.on_metrics_calculated)
        worker.signals.error.connect(self.on_metrics_error)
        self.threadpool.start(worker)       

    #--------------------------------------------------------------------------
    @Slot()
    def train_from_scratch(self):  
        self.main_win.findChild(QPushButton, "startTraining").setEnabled(False)
        self.configuration = self.config_manager.get_configuration() 
        self.training_handler = TrainingEvents(self.configuration)         
  
        # send message to status bar
        self._send_message("Training FEXT Autoencoder model from scratch...") 
        # initialize worker for asynchronous loading of the dataset
        # functions that are passed to the worker will be executed in a separate thread
        self._training_worker = Worker(self.training_handler.run_training_pipeline)                            
        worker = self._training_worker

        # inject the progress signal into the worker   
        self.train_progress_bar.setValue(0)    
        worker.signals.progress.connect(self.train_progress_bar.setValue)
        worker.signals.finished.connect(self.on_train_finished)
        worker.signals.error.connect(self.on_train_error)
        self.threadpool.start(worker)    

    #--------------------------------------------------------------------------
    @Slot()
    def resume_training_from_checkpoint(self):  
        if not self.metrics:
            return None
        
        self.main_win.findChild(QPushButton, "getImgMetrics").setEnabled(False)
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.configuration)       
        # send message to status bar
        self._send_message("Calculating image dataset evaluation metrics...") 
        # initialize worker for asynchronous loading of the dataset
        # functions that are passed to the worker will be executed in a separate thread
        self._validation_worker = Worker(
            self.validation_handler.run_dataset_evaluation_pipeline, self.metrics)                
        worker = self._validation_worker

        # inject the progress signal into the worker   
        self.data_progress_bar.setValue(0)    
        worker.signals.progress.connect(self.data_progress_bar.setValue)
        worker.signals.finished.connect(self.on_metrics_calculated)
        worker.signals.error.connect(self.on_metrics_error)
        self.threadpool.start(worker)          

    #--------------------------------------------------------------------------
    @Slot(str)
    def update_JIT_backend(self, backend: str):
        self.config_manager.update_value('jit_backend', backend)

    #--------------------------------------------------------------------------
    @Slot()
    def _update_graphics_view(self):
        source = self.plot_pixmaps if self.set_plot_view.isChecked() else self.image_pixmaps
        if source:            
            raw_pix = source[self.current_fig]
            view_size = self.view.viewport().size()
            # scale images to the canvas pixel dimensions with smooth filtering
            scaled = raw_pix.scaled(
                view_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation)
            self.pixmap_item.setPixmap(scaled)
            self.scene.setSceneRect(scaled.rect())

    #--------------------------------------------------------------------------
    @Slot()
    def show_previous_figure(self):             
        if self.current_fig > 0:
            self.current_fig -= 1
            self._update_graphics_view()

    #--------------------------------------------------------------------------
    @Slot()
    def show_next_figure(self): 
        elements = len(self.plot_pixmaps) if self.set_plot_view.isChecked() \
                   else len(self.image_pixmaps)  
            
        if self.current_fig < elements - 1:
            self.current_fig += 1
            self._update_graphics_view()

    #--------------------------------------------------------------------------
    @Slot()
    def clear_figures(self):       
        self.images = []
        self.image_pixmaps = None

    # [POSITIVE OUTCOME HANDLERS]
    ###########################################################################       
    def on_metrics_calculated(self, plots):   
        self.plots.extend(plots) if plots else None
        self.plot_pixmaps = [
            self.validation_handler.convert_fig_to_qpixmap(p) for p in self.plots]
        self.current_fig = 0
        self._update_graphics_view()
        self.validation_handler.handle_success(
            self.main_win, 'Figures have been generated')
        self.main_win.findChild(QPushButton, "getImgMetrics").setEnabled(True) 

    #--------------------------------------------------------------------------
    def on_train_finished(self, session):   
        
        self.training_handler.handle_success(
            self.main_win, 'Training session is over. Model has been saved')
        self.main_win.findChild(QPushButton, "startTraining").setEnabled(True) 
        self.main_win.findChild(QPushButton, "resumeTraining").setEnabled(True)     

    # [NEGATIVE OUTCOME HANDLERS]
    ########################################################################### #    
    @Slot(tuple)
    def on_metrics_error(self, err_tb):
        self.training_handler.handle_error(self.main_win, err_tb) 
        self.main_win.findChild(QPushButton, "getImgMetrics").setEnabled(True) 


    @Slot(tuple)
    #--------------------------------------------------------------------------
    def on_train_error(self, err_tb):
        self.training_handler.handle_error(self.main_win, err_tb) 
        self.main_win.findChild(QPushButton, "startTraining").setEnabled(True) 
        self.main_win.findChild(QPushButton, "resumeTraining").setEnabled(True)

        

    
       

    
