@echo off

cd /d %~dp0
git pull
cd ../../../python_embeded

python.exe -s -m pip install scikit-build
python.exe -s -m pip install cmake moviepy
python.exe -s -m pip install llama-cpp-python==0.1.78 --force-reinstall --upgrade --no-cache-dir
rem fix for was dependency 
python.exe -s -m pip install numpy==1.25

PAUSE