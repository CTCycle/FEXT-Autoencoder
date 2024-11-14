from nicegui import ui
import copy

from FEXT.commons.utils.app.threads import InferenceThread
from FEXT.commons.constants import PROJECT_DIR, RESULTS_PATH
from FEXT.commons.logger import logger



###############################################################################
class ProgressBarWidgets:

    def __init__(self):
        self.feedback = 'instant-feedback'

    #--------------------------------------------------------------------------
    def build_progress_bar(self):
        progress_bar = ui.linear_progress(value=0).props(self.feedback)
        progress_bar.visible = False 

        return progress_bar

    #--------------------------------------------------------------------------
    def update_progress_bar(self, progress_bar, solver : InferenceThread):
            if solver.total > 0:
                progress_bar.visible = True
                progress_bar.value = solver.progress / solver.total
                if solver.progress >= solver.total:
                    progress_bar.visible = False
                    ui.notify('Data fitting is completed')
                    # reset progress bar value
                    solver.progress = 0
                    solver.total = 0
            else:
                progress_bar.visible = False


