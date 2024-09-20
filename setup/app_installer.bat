@echo off

:: [CONDA CHECK AND INSTALLATION]
echo Checking if conda or miniconda is installed...
where conda >nul 2>&1
if errorlevel 1 (
    echo Conda or Miniconda is not installed.
    goto :condainstaller
) else (
    echo Conda is already installed. Proceeding with environment setup.
    goto :start
)

:condainstaller
set /p user_input=Would you like to download and install Miniconda automatically? (Y/N): 
if /i "%user_input%"=="Y" (
    echo Downloading Miniconda installer...

    :: Using bitsadmin to download Miniconda installer
    bitsadmin /transfer "MinicondaDownload" https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe "%cd%\MinicondaInstaller.exe"
    
    if exist "MinicondaInstaller.exe" (
        echo Running Miniconda installer...
        start /wait MinicondaInstaller.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\Miniconda3

        :: Check if installation was successful
        if exist "%UserProfile%\Miniconda3\Scripts\conda.exe" (
            echo Miniconda was installed successfully.
            goto :start
        ) else (
            echo Miniconda installation failed. Please try installing it manually.
            pause
            goto :eof
        )
    ) else (
        echo Failed to download the Miniconda installer. Please try again or download manually from the following link:
        echo https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
        pause
        goto :eof
    )
) else (
    echo Miniconda installation was skipped. Please install it manually from the following link:
    echo https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
    pause
    goto :eof
)

:: [CREATION OF PYTHON ENVIRONMENT] 
:start
echo.
conda info --envs | findstr "FEXT"
if %ERRORLEVEL%==0 (
    echo Environment already exists. Activating...
    call conda activate FEXT
    goto :dependencies
) else (
    echo Creating FEXT environment...
    call conda create -n FEXT python=3.11 -y
    call conda activate FEXT
    goto :dependencies
)

:: [INSTALL DEPENDENCIES] 
:dependencies
echo.
echo STEP 2: Install python libraries and packages
call pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
call pip install tensorflow-cpu==2.17.0 keras==3.5.0
call pip install scikit-learn==1.2.2 matplotlib==3.9.0 opencv-python==4.10.0.84
call pip install numpy==1.26.4 pandas==2.2.2 openpyxl==3.1.5 tqdm==4.66.4 
call pip install jupyter==1.1.1

:: [INSTALLATION OF PYDOT/PYDOTPLUS] 
echo Installing pydot and pydotplus...
call conda install pydot=3.0.1 -y
call conda install pydotplus -y

:: [INSTALL PROJECT IN EDITABLE MODE] 
echo STEP 3: Install utils packages in editable mode
call cd .. && pip install -e . --use-pep517

:: [CLEAN CACHE] 
echo.
echo Cleaning conda and pip cache 
call conda clean --all -y
call pip cache purge

:: [SHOW LIST OF INSTALLED DEPENDENCIES] 
echo.
echo List of installed dependencies:
call conda list

echo.
echo Installation complete.
pause
exit