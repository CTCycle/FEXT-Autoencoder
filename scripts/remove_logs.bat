@echo off
REM Change directory to the resources/logs folder 
cd /d "%~dp0..\FEXT\resources\logs"

REM Remove all .log files in the resources/logs folder
del *.log /q

REM Optionally, you can add a message to confirm the action
echo All .log files have been deleted from the resources\logs folder.