@echo off
:main_menu
cls
call conda activate FEXT
if errorlevel 1 (
    echo Conda environment activation failed.
    echo.
    echo Launching installation script (Anaconda/Miniconda must be installed first!)        
    goto :install
)

echo =======================================
echo           FEXT main app
echo =======================================
echo.
echo 1. Data analysis
echo 2. Model training and evaluation
echo 3. Extract features from images
echo 4. App maintenance
echo 5. Exit and close
echo.
set /p choice="Select an option (1-5): "

if "%choice%"=="1" goto datanalysis
if "%choice%"=="2" goto ML_menu
if "%choice%"=="3" goto inference
if "%choice%"=="4" goto maintenance
if "%choice%"=="5" goto exit
echo Invalid option, try again.
pause
goto main_menu

:install
cls
echo.
echo Starting FeXT AutoEncoder installation. Please wait until completion and do not close the console!
call ../setup/app_installer.bat
goto main_menu

:datanalysis
cls
echo.
echo Starting data analysis
call jupyter lab --NotebookApp.notebook_dir='\validation\' data_validation.ipynb
pause
goto main_menu

:inference
cls
echo.
echo Starting FEXT AutoEncoder inference module
call python inference/images_encoding.py
goto main_menu

:ML_menu
cls
echo =======================================
echo           FeXT AutoEncoder ML
echo =======================================
echo.
echo 1. Train from scratch
echo 2. Train from checkpoint
echo 3. Evaluate model performances
echo 4. Back to main menu
echo.
set /p sub_choice="Select an option (1-4): "

if "%sub_choice%"=="1" goto train_fs
if "%sub_choice%"=="2" goto train_ckpt
if "%sub_choice%"=="3" goto main_menu
if "%sub_choice%"=="4" goto modeleval
echo Invalid option, try again.
pause
goto ML_menu

:train_fs
cls
python .\training\model_training.py
pause
goto ML_menu

:train_ckpt
cls
python .\training\train_from_checkpoint.py
pause
goto ML_menu

:modeleval
cls
echo.
echo Starting data analysis
call jupyter lab --NotebookApp.notebook_dir='\validation\' model_validation.ipynb
pause
goto ML_menu

:exit
cls
echo Exiting FEXT application...
pause
exit