@echo off
rem Install packages as developer mode

call conda activate FEXT && cd .. && pip install -e . --use-pep517
if errorlevel 1 (
    echo Failed to install the package in editable mode
    goto :eof
)

