@echo off
setlocal enabledelayedexpansion

:: Specify the settings file path
set settings_file=settings/launcher_configurations.ini

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Read settings from the configurations file
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
for /f "tokens=1,2 delims==" %%a in (%settings_file%) do (
    set key=%%a
    set value=%%b
    if not "!key:~0,1!"=="[" (        
        if "!key!"=="skip_CUDA_check" set skip_CUDA_check=!value!
        if "!key!"=="use_custom_environment" set use_custom_environment=!value!
        if "!key!"=="custom_env_name" set custom_env_name=!value!
    )
)

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Check if conda is installed
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Anaconda/Miniconda is not installed. Please install it manually first.
    pause
    goto exit
) else (
    echo Anaconda/Miniconda already installed. Checking python environment...
    goto :initial_check   
)

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Check if the 'FEXT' environment exists when not using a custom environment
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:initial_check
if /i "%use_custom_environment%"=="false" (
    set "env_name=FEXT"
    goto :check_environment
) else (
    echo A custom Python environment '%custom_env_name%' has been selected.
    set "env_name=%custom_env_name%"
    goto :check_environment
)

:check_environment
set "env_exists=false"
:: Loop through Conda environments to check if the specified environment exists
for /f "skip=2 tokens=1*" %%a in ('conda env list') do (
    if /i "%%a"=="%env_name%" (
        set "env_exists=true"
        goto :env_found
    )
)

:env_found
if "%env_exists%"=="true" (
    echo Python environment '%env_name%' detected.
    goto :cudacheck
) else (
    if /i "%env_name%"=="FEXT" (
        echo Running first-time installation for FEXT. Please wait until completion and do not close the console!
        call "%~dp0\..\setup\FEXT_installer.bat"
        set "custom_env_name=FEXT"
        goto :cudacheck
    ) else (
        echo Selected custom environment '%custom_env_name%' does not exist.
        echo Please select a valid environment or set use_custom_environment=false.
        pause
        exit
    )
)

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Check if NVIDIA GPU is available using nvidia-smi
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:cudacheck
if /i "%skip_CUDA_check%"=="true" (
    goto :main_menu
) else (
    nvidia-smi >nul 2>&1
    if %ERRORLEVEL%==0 (
        echo NVIDIA GPU detected. Checking CUDA version...
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        goto :main_menu
    ) else (
        echo No NVIDIA GPU detected or NVIDIA drivers are not installed.
        goto :main_menu
    )
)

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Show main menu
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:main_menu
echo.
echo =======================================
echo           FEXT AutoEncoder
echo =======================================
echo 1. Analyze image dataset
echo 2. Model training and evaluation
echo 3. Encode images
echo 4. App setup and maintenance
echo 5. Exit and close
echo =======================================
echo.
set /p choice="Select an option (1-5): "

if "%choice%"=="1" goto :datanalysis
if "%choice%"=="2" goto :ML_menu
if "%choice%"=="3" goto :inference
if "%choice%"=="4" goto :setup_menu
if "%choice%"=="5" goto exit
echo Invalid option, try again.
pause
goto :main_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Run data analysis
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:datanalysis
cls
start cmd /k "call conda activate %env_name% && jupyter notebook .\validation\image_dataset_validation.ipynb"
goto :main_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Run model inference
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:inference
cls
call conda activate %env_name% && python .\inference\images_encoding.py
goto :main_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Check if NVIDIA GPU is available using nvidia-smi
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:ML_menu
cls
echo =======================================
echo           FEXT AutoEncoder ML
echo =======================================
echo 1. Train from scratch
echo 2. Train from checkpoint
echo 3. Evaluate model performance
echo 4. Back to main menu
echo.
set /p sub_choice="Select an option (1-4): "

if "%sub_choice%"=="1" goto :train_fs
if "%sub_choice%"=="2" goto :train_ckpt
if "%sub_choice%"=="3" goto :modeleval
if "%sub_choice%"=="4" goto :main_menu
echo Invalid option, try again.
pause
goto :ML_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Run model training from scratch
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:train_fs
cls
call conda activate %env_name% && python .\training\model_training.py
pause
goto :ML_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Run model training from checkpoint
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:train_ckpt
cls
call conda activate %env_name% && python .\training\train_from_checkpoint.py
goto :ML_menu


:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Run model evaluation
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:modeleval
cls
start cmd /k "call conda activate %env_name% && jupyter notebook .\validation\model_validation.ipynb"
goto :ML_menu

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Show setup menu
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:setup_menu
cls
echo =======================================
echo         FEXT AutoEncoder setup
echo =======================================
echo 1. Install project into environment
echo 2. Remove logs
echo 3. Back to main menu
echo.
set /p sub_choice="Select an option (1-3): "

if "%sub_choice%"=="1" goto :eggs
if "%sub_choice%"=="2" goto :logs
if "%sub_choice%"=="3" goto :main_menu
echo Invalid option, try again.
pause
goto :setup_menu

:eggs
call conda activate %env_name% && cd .. && pip install -e . --use-pep517 && cd FEXT
pause
goto :setup_menu

:logs
cd /d "%~dp0..\FEXT\resources\logs"
del *.log /q
cd /d "%~dp0..\FEXT"
pause
goto :setup_menu
