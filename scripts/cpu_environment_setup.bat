@echo off
rem Use this script to create a new environment called "FEXT"

echo STEP 1: Creation of FEXT environment
call conda create -n FEXT python=3.10 -y
if errorlevel 1 (
    echo Failed to create the environment FEXT
    goto :eof
)

rem If present, activate the environment
call conda activate FEXT

rem Install additional packages with pip
echo STEP 2: Install python libraries and packages
call pip install numpy pandas scikit-learn matplotlib python-opencv tensorflow==2.10 
if errorlevel 1 (
    echo Failed to install Python libraries.
    goto :eof
)

rem Install additional tools
echo STEP 3: Install additional libraries for model visualization
call conda install graphviz -y
call pip install pydot
if errorlevel 1 (
    echo Failed to install Graphviz or Pydot.
    goto :eof
)

@echo off
rem install packages in editable mode
echo STEP 4: Install utils packages in editable mode
call cd .. && pip install -e .
if errorlevel 1 (
    echo Failed to install the package in editable mode
    goto :eof
)

rem Clean cache
echo Cleaning conda and pip cache 
call conda clean -all -y
call pip cache purge

rem Print the list of dependencies installed in the environment
echo List of installed dependencies
call conda list

set/p<nul =Press any key to exit... & pause>nul
