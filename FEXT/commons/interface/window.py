from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, Slot, QThreadPool, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QPushButton, QRadioButton, QCheckBox, QDoubleSpinBox, 
                               QSpinBox, QComboBox, QProgressBar, QGraphicsScene, 
                               QGraphicsPixmapItem, QGraphicsView)


from FEXT.commons.variables import EnvironmentVariables
from FEXT.commons.configurations import Configurations
from FEXT.commons.interface.events import ValidationEvents, TrainingEvents
from FEXT.commons.interface.workers import Worker
from FEXT.commons.constants import UI_PATH
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
        self.config_manager = Configurations()
        self.configurations = self.config_manager.get_configurations()
    
        self.threadpool = QThreadPool.globalInstance()
        self._validation_worker = None
        self._training_worker = None        

        # get Hugging Face access token
        EV = EnvironmentVariables()
        self.env_variables = EV.get_environment_variables()

        # --- Create persistent handlers ---
        # These objects will live as long as the MainWindow instance lives
        self.validation_handler = ValidationEvents(self.configurations)
        self.training_handler = TrainingEvents(self.configurations)            

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

        # setup UI elements
        self._set_states()
        self.widgets = {}
        self._setup_configurations([
            (QCheckBox,'imgAugment','set_img_augmentation'),
            (QCheckBox,'setShuffle','set_shuffle'),
            (QCheckBox,'mixedPrecision','set_mixed_precision'),
            (QCheckBox,'compileJIT','set_JIT_compiler'),
            (QCheckBox,'runTensorboard','set_tensorboard'),
            (QCheckBox,'realTimeHistory','set_real_time_history'),
            (QCheckBox,'saveCheckpoints','set_checkpoints'),
            (QCheckBox,'useScheduler','set_LR_scheduler'),
            (QCheckBox,'getStatsAnalysis','get_image_stats'),
            (QCheckBox,'getPixDist','get_pixels_dist'),
            (QSpinBox,'seed','set_general_seed'),
            (QSpinBox,'splitSeed','set_split_seed'),
            (QSpinBox,'trainSeed','set_train_seed'),
            (QSpinBox,'shuffleSize','set_shuffle_size'),
            (QSpinBox,'numEpochs','set_epochs'),
            (QSpinBox,'batchSize','set_batch_size'),
            (QSpinBox,'deviceID','set_device_ID'),
            (QSpinBox,'saveCPFrequency','save_cp_frequency'),
            (QSpinBox,'numWorkers','set_num_workers'),
            (QSpinBox,'constantSteps','set_constant_steps'),
            (QSpinBox,'decaySteps','set_decay_steps'),
            (QDoubleSpinBox,'sampleSize','set_sample_size'),
            (QDoubleSpinBox,'trainSampleSize','set_train_sample_size'),
            (QDoubleSpinBox,'validationSize','set_validation_size'),
            (QDoubleSpinBox,'initialLearningRate','set_initial_LR'),
            (QDoubleSpinBox,'targetLearningRate','set_target_LR'),
            (QRadioButton,'setCPU','set_CPU'),
            (QRadioButton,'setGPU','set_GPU'),
            (QRadioButton,'viewPlots','set_plot_view'),
            (QRadioButton,'viewImages','set_image_view'),
            (QProgressBar,'dataProgressBar','data_progress_bar'),
            (QProgressBar,'trainingProgressBar','train_progress_bar'),
            (QComboBox,'backendJIT','combo_backend_jit'),
            (QPushButton,'getImgMetrics','btn_get_img_metrics'),
            (QPushButton,'previousImg','btn_prev_img'),
            (QPushButton,'nextImg','btn_next_img'),
            (QPushButton,'clearImg','btn_clear_img'),
            (QPushButton,'startTraining','btn_start_training'),
            (QPushButton,'resumeTraining','btn_resume_training')])
        
        self._connect_signals([
            ('set_img_augmentation','toggled',self._update_settings),
            ('set_shuffle','toggled',self._update_settings),            
            ('set_mixed_precision','toggled',self._update_settings),
            ('set_JIT_compiler','toggled',self._update_settings),
            ('set_tensorboard','toggled',self._update_settings),
            ('set_real_time_history','toggled',self._update_settings),
            ('set_checkpoints','toggled',self._update_settings),
            ('set_LR_scheduler','toggled',self._update_settings),
            ('get_image_stats','toggled',self._update_settings),
            ('get_pixels_dist','toggled',self._update_settings),
            ('set_CPU','toggled',self._update_settings),
            ('set_GPU','toggled',self._update_settings),
            ('set_general_seed','valueChanged',self._update_settings),
            ('set_split_seed','valueChanged',self._update_settings),
            ('set_train_seed','valueChanged',self._update_settings),
            ('set_shuffle_size','valueChanged',self._update_settings),            
            ('set_epochs','valueChanged',self._update_settings),
            ('set_batch_size','valueChanged',self._update_settings),
            ('set_device_ID','valueChanged',self._update_settings),
            ('save_cp_frequency','valueChanged',self._update_settings),
            ('set_num_workers','valueChanged',self._update_settings),
            ('set_constant_steps','valueChanged',self._update_settings),
            ('set_decay_steps','valueChanged',self._update_settings),
            ('set_sample_size','valueChanged',self._update_settings),
            ('set_train_sample_size','valueChanged',self._update_settings),
            ('set_validation_size','valueChanged',self._update_settings),
            ('set_initial_LR','valueChanged',self._update_settings),
            ('set_target_LR','valueChanged',self._update_settings),
            ('set_plot_view','toggled',self._update_graphics_view),
            ('set_image_view','toggled',self._update_graphics_view),
            ('combo_backend_jit','currentTextChanged',self.update_JIT_backend),
            ('btn_get_img_metrics','clicked',self.compute_image_metrics),
            ('btn_prev_img','clicked',self.show_previous_figure),
            ('btn_next_img','clicked',self.show_next_figure),
            ('btn_clear_img','clicked',self.clear_figures),
            ('btn_start_training','clicked',self.start_training),
            ('btn_resume_training','clicked',self.resume_training)])      


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
    def _setup_configurations(self, widget_defs):
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
        self.config_manager.update_value('use_img_augmentation', self.set_img_augmentation.isChecked())
        self.config_manager.update_value('shuffle_dataset', self.set_shuffle.isChecked())
        self.config_manager.update_value('num_workers', self.set_num_workers.value())
        self.config_manager.update_value('mixed_precision', self.set_mixed_precision.isChecked())
        self.config_manager.update_value('use_jit_compiler', self.set_JIT_compiler.isChecked())
        self.config_manager.update_value('run_tensorboard', self.set_tensorboard.isChecked())
        self.config_manager.update_value('real_time_history', self.set_real_time_history.isChecked())
        self.config_manager.update_value('save_checkpoints', self.set_checkpoints.isChecked())
        self.config_manager.update_value('use_lr_scheduler', self.set_LR_scheduler.isChecked())       
        self.config_manager.update_value('general_seed', self.set_general_seed.value())
        self.config_manager.update_value('split_seed', self.set_split_seed.value())
        self.config_manager.update_value('train_seed', self.set_train_seed.value())
        self.config_manager.update_value('shuffle_size', self.set_train_seed.value())
        self.config_manager.update_value('num_epochs', self.set_epochs.value())
        self.config_manager.update_value('batch_size', self.set_batch_size.value())
        self.config_manager.update_value('device_id', self.set_device_ID.value())
        self.config_manager.update_value('sample_size', self.set_sample_size.value())
        self.config_manager.update_value('train_sample_size', self.set_train_sample_size.value())
        self.config_manager.update_value('validation_size', self.set_validation_size.value())

        self.device = 'GPU' if self.set_GPU.isChecked() else 'CPU'
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
        self.configurations = self.config_manager.get_configurations() 
        self.validation_handler = ValidationEvents(self.configurations)       
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
    def start_training(self):  
        self.main_win.findChild(QPushButton, "startTraining").setEnabled(False)
        self.configurations = self.config_manager.get_configurations() 
        self.training_handler = TrainingEvents(self.configurations)         
  
        # send message to status bar
        self._send_message("Training FEXT Autoencoder model from scratch...") 
        # initialize worker for asynchronous loading of the dataset
        # functions that are passed to the worker will be executed in a separate thread
        self._validation_worker = Worker(self.training_handler.train_model)                            
        worker = self._validation_worker

        # inject the progress signal into the worker   
        self.data_progress_bar.setValue(0)    
        worker.signals.progress.connect(self.data_progress_bar.setValue)
        worker.signals.finished.connect(self.on_train_finished)
        worker.signals.error.connect(self.on_train_error)
        self.threadpool.start(worker)    

    #--------------------------------------------------------------------------
    @Slot()
    def resume_training(self):  
        if not self.metrics:
            return None
        
        self.main_win.findChild(QPushButton, "getImgMetrics").setEnabled(False)

        self.configurations = self.config_manager.get_configurations() 
        self.validation_handler = ValidationEvents(self.configurations)       
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
        elements = len(self.plot_pixmaps) if self.set_plot_view.isChecked() else len(self.image_pixmaps)      
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
    def on_train_finished(self, plots):   
        self.plots.extend(plots) if plots else None
        self.plot_pixmaps = [
            self.validation_handler.convert_fig_to_qpixmap(p) for p in self.plots]
        self.current_fig = 0
        self._update_graphics_view()
        self.validation_handler.handle_success(
            self.main_win, 'Figures have been generated')
        self.main_win.findChild(QPushButton, "getImgMetrics").setEnabled(True)     

    # [NEGATIVE OUTCOME HANDLERS]
    ########################################################################### #    
    @Slot(tuple)
    def on_metrics_error(self, err_tb):
        self.training_handler.handle_error(self.main_win, err_tb) 
        self.main_win.findChild(QPushButton, "startTraining").setEnabled(True) 
        self.main_win.findChild(QPushButton, "resumeTraining").setEnabled(True) 

    @Slot(tuple)
    #--------------------------------------------------------------------------
    def on_train_error(self, err_tb):
        self.training_handler.handle_error(self.main_win, err_tb) 
        self.main_win.findChild(QPushButton, "startTraining").setEnabled(True) 
        self.main_win.findChild(QPushButton, "resumeTraining").setEnabled(True)

        

    
       

    
