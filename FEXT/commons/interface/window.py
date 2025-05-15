from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, Slot, QThreadPool, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QPushButton, QRadioButton, QCheckBox, QDoubleSpinBox, 
                               QSpinBox, QComboBox, QProgressBar, QGraphicsScene, 
                               QGraphicsPixmapItem, QGraphicsView)


from FEXT.commons.variables import EnvironmentVariables
from FEXT.commons.configurations import Configurations
from FEXT.commons.interface.events import ValidationEvents
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
        
        # setup UI elements
        self._setup_configurations()
        self._connect_signals()
        self._set_states()

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
    def _setup_configurations(self):              
        self.set_img_augmentation = self.main_win.findChild(QCheckBox, "imgAugment")
        self.set_shuffle = self.main_win.findChild(QCheckBox, "setShuffle")
        self.set_mixed_precision = self.main_win.findChild(QCheckBox, "mixedPrecision")
        self.set_JIT_compiler = self.main_win.findChild(QCheckBox, "compileJIT")
        self.set_tensorboard = self.main_win.findChild(QCheckBox, "runTensorboard")
        self.set_real_time_history = self.main_win.findChild(QCheckBox, "realTimeHistory")
        self.set_checkpoints = self.main_win.findChild(QCheckBox, "saveCheckpoints")
        self.set_LR_scheduler = self.main_win.findChild(QCheckBox, "useScheduler")          

        self.set_general_seed = self.main_win.findChild(QSpinBox, "seed")
        self.set_split_seed = self.main_win.findChild(QSpinBox, "splitSeed")
        self.set_train_seed = self.main_win.findChild(QSpinBox, "trainSeed")
        self.set_epochs = self.main_win.findChild(QSpinBox, "numEpochs")
        self.set_batch_size = self.main_win.findChild(QSpinBox, "batchSize")
        self.set_device_ID = self.main_win.findChild(QSpinBox, "deviceID")
        self.save_cp_frequency = self.main_win.findChild(QSpinBox, "saveCPFrequency")      
        self.set_num_workers = self.main_win.findChild(QSpinBox, "numWorkers")     
        self.set_constant_steps = self.main_win.findChild(QSpinBox, "constantSteps")      
        self.set_decay_steps = self.main_win.findChild(QSpinBox, "decaySteps")  
        
        self.set_sample_size = self.main_win.findChild(QDoubleSpinBox, "sampleSize")
        self.set_train_sample_size = self.main_win.findChild(QDoubleSpinBox, "trainSampleSize")
        self.set_validation_size = self.main_win.findChild(QDoubleSpinBox, "validationSize")
        self.set_initial_LR = self.main_win.findChild(QDoubleSpinBox, "initialLearningRate")
        self.set_target_LR = self.main_win.findChild(QDoubleSpinBox, "targetLearningRate")      

        self.set_CPU = self.main_win.findChild(QRadioButton, "setCPU")
        self.set_GPU = self.main_win.findChild(QRadioButton, "setGPU")
        self.set_plot_view = self.main_win.findChild(QRadioButton, "viewPlots")
        self.set_image_view = self.main_win.findChild(QRadioButton, "viewImages")     

        # connect their toggled signals to our updater        
        self.set_img_augmentation.toggled.connect(self._update_settings)
        self.set_shuffle.toggled.connect(self._update_settings)
        self.set_mixed_precision.toggled.connect(self._update_settings)
        self.set_JIT_compiler.toggled.connect(self._update_settings)
        self.set_tensorboard.toggled.connect(self._update_settings)
        self.set_real_time_history.toggled.connect(self._update_settings)
        self.set_checkpoints.toggled.connect(self._update_settings)
        self.set_LR_scheduler.toggled.connect(self._update_settings)
        self.set_general_seed.valueChanged.connect(self._update_settings)
        self.set_split_seed.valueChanged.connect(self._update_settings)
        self.set_train_seed.valueChanged.connect(self._update_settings)
        self.set_epochs.valueChanged.connect(self._update_settings)
        self.set_batch_size.valueChanged.connect(self._update_settings)
        self.set_device_ID.valueChanged.connect(self._update_settings)
        self.save_cp_frequency.valueChanged.connect(self._update_settings)
        self.set_num_workers.valueChanged.connect(self._update_settings)
        self.set_constant_steps.valueChanged.connect(self._update_settings)
        self.set_decay_steps.valueChanged.connect(self._update_settings)
        self.set_sample_size.valueChanged.connect(self._update_settings)
        self.set_train_sample_size.valueChanged.connect(self._update_settings)
        self.set_validation_size.valueChanged.connect(self._update_settings)
        self.set_initial_LR.valueChanged.connect(self._update_settings)
        self.set_target_LR.valueChanged.connect(self._update_settings)
        self.set_CPU.toggled.connect(self._update_settings)
        self.set_GPU.toggled.connect(self._update_settings)
        self.set_plot_view.toggled.connect(self._update_graphics_view)
        self.set_image_view.toggled.connect(self._update_graphics_view)

    #--------------------------------------------------------------------------
    def _connect_signals(self):        
        self._connect_combo_box("backendJIT", self.update_JIT_backend)
        self._connect_button("imageStats", self.calculate_image_statistics)
        self._connect_button("pixelDist", self.compute_pixel_distribution_histogram)

        self._connect_button("previousImg", self.show_previous_figure)
        self._connect_button("nextImg", self.show_next_figure)    
        self._connect_button("clearImg", self.show_next_figure)    
           
       
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
        self.config_manager.update_value('mixed_precision', self.set_mixed_precision.isChecked())
        self.config_manager.update_value('use_jit_compiler', self.set_JIT_compiler.isChecked())
        self.config_manager.update_value('run_tensorboard', self.set_tensorboard.isChecked())
        self.config_manager.update_value('real_time_history', self.set_real_time_history.isChecked())
        self.config_manager.update_value('save_checkpoints', self.set_checkpoints.isChecked())
        self.config_manager.update_value('use_lr_scheduler', self.set_LR_scheduler.isChecked())       
        self.config_manager.update_value('general_seed', self.set_general_seed.value())
        self.config_manager.update_value('split_seed', self.set_split_seed.value())
        self.config_manager.update_value('train_seed', self.set_train_seed.value())
        self.config_manager.update_value('num_epochs', self.set_epochs.value())
        self.config_manager.update_value('batch_size', self.set_batch_size.value())
        self.config_manager.update_value('device_id', self.set_device_ID.value())
        self.config_manager.update_value('sample_size', self.set_sample_size.value())
        self.config_manager.update_value('train_sample_size', self.set_train_sample_size.value())
        self.config_manager.update_value('validation_size', self.set_validation_size.value())

        self.device = 'GPU' if self.set_GPU.isChecked() else 'CPU'
        self.config_manager.update_value('device', self.device)

        self.view_subject = 'plot' if self.set_plot_view.isChecked() else 'image'
        self.config_manager.update_value('view_type', self.view_subject)   


    #--------------------------------------------------------------------------
    @Slot()
    def calculate_image_statistics(self):  
        self.main_win.findChild(QPushButton, "imageStats").setEnabled(False)

        self.configurations = self.config_manager.get_configurations() 
        self.validation_handler = ValidationEvents(self.configurations) 
        
        # send message to status bar
        self._send_message("Calculating image statistics...")        

        # initialize worker for asynchronous loading of the dataset
        # functions that are passed to the worker will be executed in a separate thread
        self._validation_worker = Worker(self.validation_handler.compute_dataset_statistics)
        worker = self._validation_worker

        # inject the progress signal into the worker   
        self.data_progress_bar.setValue(0)    
        worker.signals.progress.connect(self.data_progress_bar.setValue)
        worker.signals.finished.connect(self.on_statistics_calculated)
        worker.signals.error.connect(self.on_validation_error)
        self.threadpool.start(worker)    

    #--------------------------------------------------------------------------
    @Slot()
    def compute_pixel_distribution_histogram(self): 
        self.main_win.findChild(QPushButton, "pixelDist").setEnabled(False)
        self.configurations = self.config_manager.get_configurations() 
        self.validation_handler = ValidationEvents(self.configurations)
        
        # send message to status bar
        self._send_message("Computing pixel distribution...")

        # initialize worker for asynchronous loading of the dataset
        # functions that are passed to the worker will be executed in a separate thread
        self._validation_worker = Worker(self.validation_handler.get_pixel_distribution)
        worker = self._validation_worker

        # inject the progress signal into the worker   
        self.data_progress_bar.setValue(0)    
        worker.signals.progress.connect(self.data_progress_bar.setValue)
        worker.signals.finished.connect(self.on_generated_plot)
        worker.signals.error.connect(self.on_validation_error)
        self.threadpool.start(worker)     

    #--------------------------------------------------------------------------
    @Slot(str)
    def update_JIT_backend(self, backend: str):
        self.config_manager.update_value('jit_backend', backend)

    #--------------------------------------------------------------------------
    @Slot()
    def _update_graphics_view(self):
        source = self.plot_pixmaps if self.set_plot_view.isChecked() else self.image_pixmaps
        if source is not None:            
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
    def on_statistics_calculated(self, datasets):   
        message = "Image dataset statistics have been calculated"
        self.validation_handler.handle_success(self.main_win, message) 
        self.main_win.findChild(QPushButton, "imageStats").setEnabled(True)       

    #--------------------------------------------------------------------------
    @Slot(object)    
    def on_generated_plot(self, plots):        
        self.plots.append(plots)
        self.plot_pixmaps = [self.validation_handler.convert_fig_to_qpixmap(p) for p in self.plots]
        self.current_fig = 0
        self._update_graphics_view()
        self.validation_handler.handle_success(
            self.main_win, 'Figures have been generated')
        self.main_win.findChild(QPushButton, "pixelDist").setEnabled(True)  

    # [NEGATIVE OUTCOME HANDLERS]
    ########################################################################### #    
    @Slot(tuple)
    def on_validation_error(self, err_tb):
        self.validation_handler.handle_error(self.main_win, err_tb) 
        self.main_win.findChild(QPushButton, "pixelDist").setEnabled(True) 
        self.main_win.findChild(QPushButton, "imageStats").setEnabled(True)       
 

    
       

    
