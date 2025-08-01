import traceback
import inspect
from multiprocessing import Process, Queue, Event

from PySide6.QtCore import QObject, Signal, QRunnable, Slot

from FEXT.app.constants import ROOT_DIR, DATA_PATH
from FEXT.app.logger import logger


###############################################################################
class WorkerInterrupted(Exception):
    """Exception to indicate worker was intentionally interrupted."""
    pass


###############################################################################
class WorkerSignals(QObject):
    finished = Signal(object)      
    error = Signal(tuple) 
    interrupted = Signal()
    progress = Signal(int)

    
###############################################################################
class ThreadWorker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self._is_interrupted = False 

        sig = inspect.signature(fn)
        params = sig.parameters.values()

        # Accept if it has an explicit 'progress_callback' param or **kwargs
        accepts_progress = any(
            p.name == "progress_callback" or
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in params)
        
        if accepts_progress:
            self.kwargs["progress_callback"] = self.signals.progress.emit

        # Accept if it has an explicit 'worker' param or **kwargs
        accepts_worker = any(
            p.name == "worker" or
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in params)
        
        if accepts_worker:
            self.kwargs["worker"] = self 

    #--------------------------------------------------------------------------
    def stop(self):
        self._is_interrupted = True

    #--------------------------------------------------------------------------
    def is_interrupted(self):
        return self._is_interrupted

    #--------------------------------------------------------------------------
    @Slot()    
    def run(self):
        try:
            # Remove progress_callback and worker if not accepted by the function
            if "progress_callback" in self.kwargs and \
               "progress_callback" not in inspect.signature(self.fn).parameters:
                self.kwargs.pop("progress_callback")
            if "worker" in self.kwargs and \
               "worker" not in inspect.signature(self.fn).parameters:
                self.kwargs.pop("worker")
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except WorkerInterrupted:
            self.signals.interrupted.emit()
        except Exception as e:
            tb = traceback.format_exc()
            self.signals.error.emit((e, tb))

    #--------------------------------------------------------------------------
    def cleanup(self):
        pass 



###############################################################################‗
def process_target(fn, args, kwargs, result_queue, progress_queue, interrupted_event):
    import inspect
    try:
        # Add progress_callback and worker if supported
        sig = inspect.signature(fn)
        params = sig.parameters.values()
        # Progress
        if any(p.name == "progress_callback" or p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            kwargs = dict(kwargs)  # Make a copy
            kwargs["progress_callback"] = progress_queue.put
        # Interruption
        if any(p.name == "worker" or p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            class DummyWorker:
                def is_interrupted(self2):
                    return interrupted_event.is_set()
            kwargs["worker"] = DummyWorker()
        result = fn(*args, **kwargs)
        result_queue.put(("finished", result))
    except WorkerInterrupted:
        result_queue.put(("interrupted", None))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        result_queue.put(("error", (e, tb)))

###############################################################################
class ProcessWorker(QObject):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        self._result_queue = Queue()
        self._progress_queue = Queue()
        self._interrupted = Event()
        self._proc = None
        self._timer = None  

    #--------------------------------------------------------------------------
    def stop(self):
        self._interrupted.set()

    #--------------------------------------------------------------------------
    def is_interrupted(self):
        return self._interrupted.is_set()

    #--------------------------------------------------------------------------
    def start(self):
        self._proc = Process(
        target=process_target,
        args=(self.fn, self.args, self.kwargs, self._result_queue, self._progress_queue, self._interrupted)
        )
        self._proc.start()

    #--------------------------------------------------------------------------
    def _run_in_process(self):
        try:
            # Prepare kwargs for the child process
            fn = self.fn
            args = self.args
            kwargs = self.kwargs.copy()

            sig = inspect.signature(fn)
            params = sig.parameters.values()

            # Progress
            if any(p.name == "progress_callback" or p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
                kwargs["progress_callback"] = self._progress_queue.put

            # Interruption
            if any(p.name == "worker" or p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
                class DummyWorker:
                    def is_interrupted(self2):  # "self2" to avoid confusion
                        return self._interrupted.is_set()
                kwargs["worker"] = DummyWorker()

            result = fn(*args, **kwargs)
            self._result_queue.put(("finished", result))
        except WorkerInterrupted:
            self._result_queue.put(("interrupted", None))
        except Exception as e:
            tb = traceback.format_exc()
            self._result_queue.put(("error", (e, tb)))

    #--------------------------------------------------------------------------
    def poll(self):
        # Called periodically from main thread (window.py QTimer)
        # Progress first
        while not self._progress_queue.empty():
            try:
                progress = self._progress_queue.get_nowait()
                self.signals.progress.emit(progress)
            except Exception:
                pass

        # Result/Termination
        if not self._result_queue.empty():
            msg_type, data = self._result_queue.get()
            if msg_type == "finished":
                self.signals.finished.emit(data)
            elif msg_type == "error":
                self.signals.error.emit(data)
            elif msg_type == "interrupted":
                self.signals.interrupted.emit()
            # Stop polling when result comes
            if self._timer is not None:
                self._timer.stop()
            if self._proc is not None:
                self._proc.join()

    #--------------------------------------------------------------------------
    def cleanup(self):
        if self._timer is not None:
            self._timer.stop()
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join() 

# [HELPERS FUNCTIONS]
#------------------------------------------------------------------------------
def check_thread_status(worker : ThreadWorker):
    if worker is not None and worker.is_interrupted():        
        raise WorkerInterrupted()    

#------------------------------------------------------------------------------
def update_progress_callback(progress, total, progress_callback=None):   
    if progress_callback is not None:        
        percent = int(progress * 100 / total)
        progress_callback(percent)  
