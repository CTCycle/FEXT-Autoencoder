import os
from nicegui import ui

from FEXT.commons.utils.app.threads import InferenceThread
from FEXT.commons.constants import CHECKPOINT_PATH
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


###############################################################################
class CheckpointLoadingWidgets:
     
    def __init__(self):
        self.feedback = 'instant-feedback'
        self.selected_checkpoint = None  # Store the select component for later access

    #--------------------------------------------------------------------------
    def build_checkpoints_selector(self):

        models_list = self.update_checkpoints_list()
        ui.label('Available checkpoints')
        
        with ui.row().classes('w-full no-wrap'):            
            ui.button(icon='refresh', on_click=self.on_refresh_click).classes('icon-button')           
            self.selected_checkpoint = ui.select(models_list, label='Select checkpoints').classes('w-full')

        return self.selected_checkpoint

    #--------------------------------------------------------------------------
    def update_checkpoints_list(self):
          
        models_list = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():
                models_list.append(entry.name) 

        models_list.sort()       
        
        if not models_list:
            logger.error('No pretrained model checkpoints in resources')

        return models_list

    #--------------------------------------------------------------------------
    def on_refresh_click(self, e):
        # Update the models list
        models_list = self.update_checkpoints_list()
        # Update the options of the select component
        self.selected_checkpoint.options = models_list
        # Refresh the UI to display the updated options
        self.selected_checkpoint.update()