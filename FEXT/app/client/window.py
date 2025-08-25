from FEXT.app.variables import EnvironmentVariables
EV = EnvironmentVariables()

from functools import partial
from qt_material import apply_stylesheet
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, Slot, QThreadPool, QTimer, Qt
from PySide6.QtGui import QPainter, QPixmap, QAction
from PySide6.QtWidgets import (QPushButton, QRadioButton, QCheckBox, QDoubleSpinBox, 
                               QSpinBox, QComboBox, QProgressBar, QGraphicsScene, QGraphicsPixmapItem, 
                               QGraphicsView, QMessageBox, QDialog, QApplication)

from FEXT.app.utils.data.database import database
from FEXT.app.configuration import Configuration
from FEXT.app.client.dialogs import SaveConfigDialog, LoadConfigDialog
from FEXT.app.client.events import GraphicsHandler, ValidationEvents, ModelEvents
from FEXT.app.client.workers import ThreadWorker, ProcessWorker
from FEXT.app.constants import IMG_PATH, INFERENCE_INPUT_PATH
from FEXT.app.logger import logger


###############################################################################
def apply_style(app : QApplication):
    theme = 'dark_yellow'
    extra = {'density_scale': '-1'}
    apply_stylesheet(app, theme=f'{theme}.xml', extra=extra)

    # Make % text visible/centered for ALL progress bars
    app.setStyleSheet(app.styleSheet() + """
    QProgressBar {
        text-align: center;   /* align percentage to the center */
        color: black;        /* black text for yellow bar */
        font-weight: bold;   /* bold percentage */        
    }
    """)

    return app



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
        self.threadpool.setExpiryTimeout(2000)
        self.worker = None        

        # initialize database        
        database.initialize_database()          

        # --- Create persistent handlers ---
        self.graphic_handler = GraphicsHandler()
        self.validation_handler = ValidationEvents(self.configuration)
        self.model_handler = ModelEvents(self.configuration)        

        # setup UI elements
        self._set_states()
        self.widgets = {}
        self._setup_configuration([ 
            # actions
            (QAction, 'actionLoadConfig', 'load_configuration_action'),
            (QAction, 'actionSaveConfig', 'save_configuration_action'),
            (QAction, 'actionDeleteData', 'delete_data_action'),
            (QAction, 'actionExportData', 'export_data_action'),
            # out of tab widgets            
            (QProgressBar,'progressBar','progress_bar'),      
            (QPushButton,'stopThread','stop_thread'),
            # 1. data tab page 
            # dataset evaluation group 
            (QDoubleSpinBox,'sampleSize','sample_size'),
            (QSpinBox,'seed','seed'),                      
            (QCheckBox,'imgStatistics','img_statistics_metric'),      
            (QCheckBox,'imgExposure','img_exposure_metric'),
            (QCheckBox,'imgColor','img_color_metric'),  
            (QCheckBox,'imgEntropy','img_entropy_metric'),      
            (QCheckBox,'imgEdges','img_edges_metric'),
            (QCheckBox,'imgSharpness','img_sharpness_metric'),           
            (QCheckBox,'textureLTB','img_texture_metric'), 
            (QCheckBox,'pixDist','pixel_distribution_metric'),
            (QPushButton,'evaluateDataset','evaluate_dataset'), 
            # 2. training tab page
            # dataset settings group 
            (QCheckBox,'grayScale','use_grayscale'),
            (QCheckBox,'imgAugment','img_augmentation'), 
            (QSpinBox,'imgHeight','image_height'),
            (QSpinBox,'imgWidth','image_width'),   
            (QDoubleSpinBox,'trainSampleSize','train_sample_size'),
            (QDoubleSpinBox,'validationSize','validation_size'),
            (QSpinBox,'splitSeed','split_seed'),
            (QCheckBox,'setShuffle','use_shuffle'),
            (QSpinBox,'shuffleSize','shuffle_size'),            
            # training settings group
            (QCheckBox,'mixedPrecision','use_mixed_precision'),
            (QCheckBox,'compileJIT','use_JIT_compiler'),   
            (QComboBox,'backendJIT','jit_backend'), 
            (QSpinBox,'trainSeed','train_seed'),           
            (QSpinBox,'numEpochs','epochs'),
            (QSpinBox,'batchSize','batch_size'),   
            (QCheckBox,'useScheduler','LR_scheduler'), 
            (QDoubleSpinBox,'initialLearningRate','initial_LR'),
            (QDoubleSpinBox,'targetLearningRate','target_LR'),            
            (QSpinBox,'constantSteps','constant_steps'),
            (QSpinBox,'decaySteps','decay_steps'),  
            (QCheckBox,'runTensorboard','use_tensorboard'),
            (QCheckBox,'realTimeHistory','real_time_history_callback'),
            (QCheckBox,'saveCheckpoints','save_checkpoints'),
            (QSpinBox,'saveCPFrequency','checkpoints_frequency'),         
            # model settings group
            (QComboBox,'modelType','selected_model'), 
            (QDoubleSpinBox,'dropoutRate','dropout_rate'),
            # session settings group   
            (QCheckBox,'deviceGPU','use_device_GPU'),         
            (QSpinBox,'deviceID','device_ID'),
            (QSpinBox,'numWorkers','num_workers'),         
            (QSpinBox,'numAdditionalEpochs','additional_epochs'),                      
            (QPushButton,'startTraining','start_training'),
            (QPushButton,'resumeTraining','resume_training'),            
            # 3. model inference and evaluation tab 
            (QPushButton,'refreshCheckpoints','refresh_checkpoints'),
            (QComboBox,'checkpointsList','checkpoints_list'),
            (QPushButton,'evaluateModel','model_evaluation'),             
            (QPushButton,'checkpointSummary','checkpoints_summary'),
            (QCheckBox,'evalReport','get_evaluation_report'), 
            (QCheckBox,'imgReconstruction','image_reconstruction'), 
            (QSpinBox,'inferenceBatchSize','inference_batch_size'),     
            (QSpinBox,'numImages','num_evaluation_images'), 
            (QPushButton,'encodeImages','encode_images'),          
            # 5. Viewer tab
            (QPushButton,'loadImages','load_source_images'),
            (QPushButton,'previousImg','previous_image'),
            (QPushButton,'nextImg','next_image'),
            (QPushButton,'clearImg','clear_images'),
            (QRadioButton,'viewInferenceImages','inference_img_view'),
            (QRadioButton,'viewTrainImages','train_img_view'),
            ])
        
        self._connect_signals([ 
            # actions
            ('save_configuration_action', 'triggered', self.save_configuration),   
            ('load_configuration_action', 'triggered', self.load_configuration),
            ('delete_data_action', 'triggered', self.delete_all_data),   
            ('export_data_action', 'triggered', self.export_all_data),
            # out of tab widgets    
            ('stop_thread','clicked',self.stop_running_worker),          
            # 1. data tab page                      
            ('img_statistics_metric','toggled',self._update_metrics),
            ('pixel_distribution_metric','toggled',self._update_metrics),
            ('evaluate_dataset','clicked',self.run_dataset_evaluation_pipeline),           
            # 2. training tab page               
            ('start_training','clicked',self.train_from_scratch),
            ('resume_training','clicked',self.resume_training_from_checkpoint),
            # 3. model inference and evaluation tab page
            ('checkpoints_list','currentTextChanged',self.select_checkpoint), 
            ('refresh_checkpoints','clicked',self.load_checkpoints), 
            ('image_reconstruction','toggled',self._update_metrics),
            ('get_evaluation_report','toggled',self._update_metrics),            
            ('model_evaluation','clicked', self.run_model_evaluation_pipeline),
            ('checkpoints_summary','clicked',self.get_checkpoints_summary),                                      
            ('encode_images','clicked',self.encode_img_with_checkpoint),            
            # 4. viewer tab page             
            ('inference_img_view', 'toggled', self._update_graphics_view), 
            ('train_img_view', 'toggled', self._update_graphics_view), 
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
            # 1. data tab page
            # dataset evaluation group
            ('seed', 'valueChanged', 'seed'),
            ('sample_size', 'valueChanged', 'sample_size'),            
            # 2. training tab page
            # dataset settings group  
            ('image_height', 'valueChanged', 'image_height'),
            ('image_width', 'valueChanged', 'image_width'),
            ('use_grayscale', 'toggled', 'use_grayscale'),          
            ('img_augmentation', 'toggled', 'img_augmentation'),
            ('use_shuffle', 'toggled', 'shuffle_dataset'),
            ('shuffle_size', 'valueChanged', 'shuffle_size'),
            ('train_sample_size', 'valueChanged', 'train_sample_size'),  
            ('validation_size', 'valueChanged', 'validation_size'),    
            ('split_seed', 'valueChanged', 'split_seed'),           
            # device settings group
            ('device_ID', 'valueChanged', 'device_id'),
            ('num_workers', 'valueChanged', 'num_workers'),
            # training settings group
            ('use_tensorboard', 'toggled', 'use_tensorboard'),
            ('real_time_history_callback', 'toggled', 'real_time_history_callback'),
            ('save_checkpoints', 'toggled', 'save_checkpoints'),
            ('checkpoints_frequency', 'valueChanged', 'checkpoints_frequency'),
            ('epochs', 'valueChanged', 'epochs'),
            ('batch_size', 'valueChanged', 'batch_size'),
            ('train_seed', 'valueChanged', 'train_seed'),
            # RL scheduler settings group
            ('LR_scheduler', 'toggled', 'use_LR_scheduler'),
            ('initial_LR', 'valueChanged', 'initial_LR'),
            ('target_LR', 'valueChanged', 'target_LR'),
            ('constant_steps', 'valueChanged', 'constant_steps'),
            ('decay_steps', 'valueChanged', 'decay_steps'),
            # model settings group
            ('selected_model', 'currentTextChanged', 'selected_model'),
            ('use_mixed_precision', 'toggled', 'mixed_precision'),
            ('use_JIT_compiler', 'toggled', 'use_jit_compiler'),
            ('jit_backend', 'currentTextChanged', 'jit_backend'),
            ('dropout_rate', 'valueChanged', 'dropout_rate'),
            # session settings group
            ('additional_epochs', 'valueChanged', 'additional_epochs'),
            # 3. model evaluation and inference tab page            
            ('inference_batch_size', 'valueChanged', 'inference_batch_size'),
            ('num_evaluation_images', 'valueChanged', 'num_evaluation_images'), 
            ]

        self.data_metrics = [('image_statistics', self.img_statistics_metric), 
                             ('image_exposure', self.img_exposure_metric),
                             ('image_entropy', self.img_entropy_metric),   
                             ('image_colorimetry', self.img_color_metric),                         
                             ('image_edges', self.img_edges_metric),
                             ('image_sharpness', self.img_sharpness_metric),
                             ('image_texture', self.img_texture_metric),
                             ('pixels_distribution', self.pixel_distribution_metric)]
        self.model_metrics = [('evaluation_report', self.get_evaluation_report),
                              ('image_reconstruction', self.image_reconstruction)]                

        for attr, signal_name, config_key in connections:
            widget = self.widgets[attr]
            self.connect_update_setting(widget, signal_name, config_key)

    #--------------------------------------------------------------------------
    def _set_states(self):         
        self.progress_bar = self.main_win.findChild(QProgressBar, "progressBar")        
        self.progress_bar.setValue(0)

    #--------------------------------------------------------------------------
    def get_current_pixmaps_key(self):
        for radio, idx_key in self.pixmap_sources.items():
            if radio.isChecked():
                return self.pixmaps[idx_key], idx_key
        return [], None 

    #--------------------------------------------------------------------------
    def _set_graphics(self):      
        view = self.main_win.findChild(QGraphicsView, 'canvas')
        scene = QGraphicsScene()
        pixmap_item = QGraphicsPixmapItem()
        pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        scene.addItem(pixmap_item)
        view.setScene(scene)
        for hint in (QPainter.Antialiasing, QPainter.SmoothPixmapTransform, 
                     QPainter.TextAntialiasing):
            view.setRenderHint(hint, True)

        self.graphics = {'view': view, 'scene': scene, 'pixmap_item': pixmap_item}
        self.pixmaps = {k: [] for k in ('train_images', 'inference_images')}
        self.img_paths = {'train_images': IMG_PATH, 'inference_images': INFERENCE_INPUT_PATH}
        self.current_fig = {k: 0 for k in self.pixmaps}

        self.pixmap_sources = {self.inference_img_view: "inference_images",
                               self.train_img_view: "train_images"}        

    #--------------------------------------------------------------------------
    def _connect_button(self, button_name: str, slot):        
        button = self.main_win.findChild(QPushButton, button_name)
        button.clicked.connect(slot) 

    #--------------------------------------------------------------------------
    def _connect_combo_box(self, combo_name: str, slot):        
        combo = self.main_win.findChild(QComboBox, combo_name)
        combo.currentTextChanged.connect(slot)

    #--------------------------------------------------------------------------
    def _start_thread_worker(self, worker : ThreadWorker, on_finished, on_error, on_interrupted,
                      update_progress=True):
        if update_progress:       
            self.progress_bar.setValue(0)
            worker.signals.progress.connect(self.progress_bar.setValue)

        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(on_error)        
        worker.signals.interrupted.connect(on_interrupted)
        self.threadpool.start(worker)  

    #--------------------------------------------------------------------------
    def _start_process_worker(self, worker : ProcessWorker, on_finished, on_error, 
                              on_interrupted, update_progress=True):
        if update_progress:
            self.progress_bar.setValue(0)
            worker.signals.progress.connect(self.progress_bar.setValue)

        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(on_error)
        worker.signals.interrupted.connect(on_interrupted)

        # Polling for results from the process queue
        self.process_worker_timer = QTimer()
        self.process_worker_timer.setInterval(100)  # Check every 100ms
        self.process_worker_timer.timeout.connect(worker.poll)
        worker._timer = self.process_worker_timer
        self.process_worker_timer.start()

        worker.start()  
   
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
    def _set_widgets_from_configuration(self):
        cfg = self.config_manager.get_configuration()
        for attr, widget in self.widgets.items():
            if attr not in cfg:
                continue
            v = cfg[attr]
            # CheckBox
            if hasattr(widget, "setChecked") and isinstance(v, bool):
                widget.setChecked(v)
            # Numeric widgets (SpinBox/DoubleSpinBox)
            elif hasattr(widget, "setValue") and isinstance(v, (int, float)):
                widget.setValue(v)
            # PlainTextEdit/TextEdit
            elif hasattr(widget, "setPlainText") and isinstance(v, str):
                widget.setPlainText(v)
            # LineEdit (or any widget with setText)
            elif hasattr(widget, "setText") and isinstance(v, str):
                widget.setText(v) 
   
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
    def _update_metrics(self):             
        self.selected_metrics['dataset'] = [
            name for name, box in self.data_metrics if box.isChecked()]
        self.selected_metrics['model'] = [
            name for name, box in self.model_metrics if box.isChecked()]

    #--------------------------------------------------------------------------
    # [ACTIONS]
    #--------------------------------------------------------------------------
    @Slot()
    def save_configuration(self):
        dialog = SaveConfigDialog(self.main_win)
        if dialog.exec() == QDialog.Accepted:
            name = dialog.get_name()
            name = 'default_config' if not name else name            
            self.config_manager.save_configuration_to_json(name)
            self._send_message(f"Configuration [{name}] has been saved")

    #--------------------------------------------------------------------------
    @Slot()
    def load_configuration(self):
        dialog = LoadConfigDialog(self.main_win)
        if dialog.exec() == QDialog.Accepted:
            name = dialog.get_selected_config()
            self.config_manager.load_configuration_from_json(name)                
            self._set_widgets_from_configuration()
            self._send_message(f"Loaded configuration [{name}]")

    #--------------------------------------------------------------------------
    @Slot()
    def export_all_data(self):
        database.export_all_tables_as_csv()
        message = 'All data from database has been exported'
        logger.info(message)
        self._send_message(message)

    #--------------------------------------------------------------------------
    @Slot()
    def delete_all_data(self):      
        database.delete_all_data()        
        message = 'All data from database has been deleted'
        logger.info(message)
        self._send_message(message)

    #--------------------------------------------------------------------------
    # [GRAPHICS]
    #--------------------------------------------------------------------------
    @Slot(str)
    def _update_graphics_view(self):  
        pixmaps, idx_key = self.get_current_pixmaps_key()
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
        pixmaps, idx_key = self.get_current_pixmaps_key()
        if not pixmaps or idx_key is None:
            return
        if self.current_fig[idx_key] > 0:
            self.current_fig[idx_key] -= 1
            self._update_graphics_view()

    #--------------------------------------------------------------------------
    @Slot(str)
    def show_next_figure(self):
        pixmaps, idx_key = self.get_current_pixmaps_key()
        if not pixmaps or idx_key is None:
            return
        if self.current_fig[idx_key] < len(pixmaps) - 1:
            self.current_fig[idx_key] += 1
            self._update_graphics_view()

    #--------------------------------------------------------------------------
    @Slot(str)
    def clear_figures(self):
        pixmaps, idx_key = self.get_current_pixmaps_key()
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
        pixmaps, idx_key = self.get_current_pixmaps_key()
        if idx_key not in self.img_paths.keys():
            return
        
        self.pixmaps[idx_key].clear()
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.configuration)
        
        img_paths = self.validation_handler.load_img_path(self.img_paths[idx_key])
        self.pixmaps[idx_key].extend(img_paths)
        self.current_fig[idx_key] = 0 
        self._update_graphics_view()   

    #--------------------------------------------------------------------------
    # [DATASET TAB]
    #--------------------------------------------------------------------------        
    @Slot()
    def run_dataset_evaluation_pipeline(self):   
        if self.worker:            
            message = "A task is currently running, wait for it to finish and then try again"
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return 
                
        if not self.selected_metrics['dataset']:
            return
        
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.configuration)       
        # send message to status bar
        self._send_message("Calculating image dataset evaluation metrics...") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ThreadWorker(
            self.validation_handler.run_dataset_evaluation_pipeline,
            self.selected_metrics['dataset'])   

        # start worker and inject signals
        self._start_thread_worker(
            self.worker, on_finished=self.on_dataset_evaluation_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted)       

    #--------------------------------------------------------------------------
    # [TRAINING TAB]
    #-------------------------------------------------------------------------- 
    @Slot()
    def train_from_scratch(self):
        if self.worker:            
            message = "A task is currently running, wait for it to finish and then try again"
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return 
                  
        self.configuration = self.config_manager.get_configuration() 
        self.model_handler = ModelEvents(self.configuration)         
  
        # send message to status bar
        self._send_message("Training FEXT Autoencoder using a new model instance...")        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ProcessWorker(self.model_handler.run_training_pipeline)                            
       
        # start worker and inject signals
        self._start_process_worker(
            self.worker, on_finished=self.on_train_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted)  

    #--------------------------------------------------------------------------
    @Slot()
    def resume_training_from_checkpoint(self): 
        if self.worker:            
            message = "A task is currently running, wait for it to finish and then try again"
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return    

        if not self.selected_checkpoint:
            return    
              
        self.configuration = self.config_manager.get_configuration() 
        self.model_handler = ModelEvents(self.configuration)   

        # send message to status bar
        self._send_message(f"Resume training from checkpoint {self.selected_checkpoint}")         
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ProcessWorker(
            self.model_handler.resume_training_pipeline,            
            self.selected_checkpoint)   

        # start worker and inject signals
        self._start_process_worker(
            self.worker, on_finished=self.on_train_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted)

    #--------------------------------------------------------------------------
    # [MODEL EVALUATION AND INFERENCE TAB]
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
    def run_model_evaluation_pipeline(self):          
        if self.worker:            
            message = "A task is currently running, wait for it to finish and then try again"
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return 
        
        if not self.selected_metrics['model'] or not self.selected_checkpoint:
            return

        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.configuration)         
        # send message to status bar
        self._send_message(f"Evaluating {self.selected_checkpoint} performances... ")

        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ProcessWorker(
            self.validation_handler.run_model_evaluation_pipeline,
            self.selected_metrics['model'], 
            self.selected_checkpoint)                
        
        # start worker and inject signals
        self._start_process_worker(
            self.worker, on_finished=self.on_model_evaluation_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted)     

    #-------------------------------------------------------------------------- 
    @Slot()
    def get_checkpoints_summary(self):       
        if self.worker:            
            message = "A task is currently running, wait for it to finish and then try again"
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return 
        
        self.configuration = self.config_manager.get_configuration() 
        self.validation_handler = ValidationEvents(self.configuration)           
        # send message to status bar
        self._send_message("Generating checkpoints summary...") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ThreadWorker(self.validation_handler.get_checkpoints_summary) 

        # start worker and inject signals
        self._start_thread_worker(
            self.worker, on_finished=self.on_model_evaluation_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted)  

    #--------------------------------------------------------------------------
    # [INFERENCE TAB]
    #--------------------------------------------------------------------------   
    @Slot()    
    def encode_img_with_checkpoint(self):  
        if self.worker:            
            message = "A task is currently running, wait for it to finish and then try again"
            QMessageBox.warning(self.main_win, "Application is still busy", message)
            return 
        
        if not self.selected_checkpoint:            
            return 
        
        self.configuration = self.config_manager.get_configuration() 
        self.model_handler = ModelEvents(self.configuration)            
        # send message to status bar
        self._send_message(f"Encoding images with {self.selected_checkpoint}") 
        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = ProcessWorker(
            self.model_handler.run_inference_pipeline,
            self.selected_checkpoint)

        # start worker and inject signals
        self._start_process_worker(
            self.worker, on_finished=self.on_inference_finished,
            on_error=self.on_error,
            on_interrupted=self.on_task_interrupted)


    ###########################################################################
    # [POSITIVE OUTCOME HANDLERS]
    ###########################################################################       
    def on_dataset_evaluation_finished(self, plots):
        self._send_message('Figures have been generated')
        self.worker = self.worker.cleanup()
        
    #--------------------------------------------------------------------------
    def on_train_finished(self, session):          
        self._send_message('Training session is over. Model has been saved')
        self.worker = self.worker.cleanup()
      
    #--------------------------------------------------------------------------
    def on_model_evaluation_finished(self, plots):  
        self._send_message(f'Model {self.selected_checkpoint} has been evaluated')
        self.worker = self.worker.cleanup()

    #--------------------------------------------------------------------------
    def on_inference_finished(self, session):          
        self._send_message('Inference call has been terminated')
        self.worker = self.worker.cleanup()


    ###########################################################################   
    # [NEGATIVE OUTCOME HANDLERS]
    ########################################################################### 
    def on_error(self, err_tb):
        exc, tb = err_tb
        logger.error(f"{exc}\n{tb}")
        message = "An error occurred during the operation. Check the logs for details."
        QMessageBox.critical(self.main_win, 'Something went wrong!', message)
        self.progress_bar.setValue(0)      
        self.worker = self.worker.cleanup()  

    ###########################################################################   
    # [INTERRUPTION HANDLERS]
    ###########################################################################     
    def on_task_interrupted(self):         
        self.progress_bar.setValue(0)        
        self._send_message('Current task has been interrupted by user')
        logger.warning('Current task has been interrupted by user')                
        self.worker = self.worker.cleanup()
        


    
       

    
