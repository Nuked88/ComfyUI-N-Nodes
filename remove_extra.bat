@echo off
set "python_exec=..\..\..\python_embeded\python.exe"

echo Restore original ComfyUI dependency...
if exist "%python_exec%" (
    echo Restore with ComfyUI Portable
    "%python_exec%" -s -m pip install transformers==4.26.1
    echo Done. Please reboot ComfyUI.
) else (
    echo Restore with system Python
    pip install transformers==4.26.1
    echo Done. Please reboot ComfyUI.
)

pause
