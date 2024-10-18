@echo off
setlocal enabledelayedexpansion

:: [INITIALIZE VARIABLES] 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Set initial state and variables
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
set "empty=1"
set /a i=0

:: [CHECK FOR TENSORBOARD FOLDERS]
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Loop through each subfolder and check if 'tensorboard' subfolder exists.
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
echo List of available checkpoints
echo.
for /d %%D in (*) do (
    if exist "%%D\tensorboard" (
        set /a i+=1
        echo [!i!] %%D
        set "folder[!i!]=%%D"
        set "empty=0"
    )
)

:: [CHECK IF NO VALID FOLDERS FOUND]
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: If no folder with a 'tensorboard' subfolder was found, display a message
:: and exit the script.
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
if "!empty!"=="1" (
    echo No checkpoints with tensorboard detected in training/checkpoints.
    echo Please be sure to add a folder containing a tensorboard subfolder and try again.
    pause
    exit /b
)

:: [DISPLAY AVAILABLE FOLDERS]
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: List only the folders that contain a 'tensorboard' subfolder for user selection.
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
echo.
echo List of available checkpoints with tensorboard logs:
echo.
set /a i=0
for /d %%D in (*) do (
    if exist "%%D\tensorboard" (
        set /a i+=1
        echo [!i!] %%D
        set "folder[!i!]=%%D"
    )
)

:: [USER SELECTION]
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Prompt the user to select a folder from the displayed list.
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
echo.
set /p foldernum=Select a checkpoint folder: 
set selected_folder=!folder[%foldernum%]!

if not defined selected_folder (
    echo Invalid selection. Exiting.
    exit /b
)

:: [NAVIGATE TO SELECTED FOLDER AND RUN TENSORBOARD]
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Navigate to the selected folder and run TensorBoard using the log directory.
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
cd "!selected_folder!"
call conda activate %env_name% && python -m tensorboard.main --logdir tensorboard/
pause
