@echo off
set "python_exec=..\..\..\python_embeded\python.exe"

echo Installing dependency for moondream_repo...
if exist "%python_exec%" (
    echo Installing with ComfyUI Portable
    "%python_exec%" -s -m pip install transformers==4.36.2
    echo Done. Please reboot ComfyUI.
) else (
    echo Installing with system Python
    pip install  transformers==4.36.2
    echo Done. Please reboot ComfyUI.
)

pause
