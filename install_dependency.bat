K:
cd K:\VARIE\ComfyUI_windows_portable\python_embeded

SET CMAKE_ARGS=-DLLAMA_CUBLAS=on
SET FORCE_CMAKE=1 
SET CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin;%PATH%
xcopy "%CUDA_PATH%\extras\visual_studio_integration\MSBuildExtensions\" "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations\" /E /I /Y
xcopy "%CUDA_PATH%\extras\visual_studio_integration\MSBuildExtensions\" "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations\" /E /I /Y


python.exe -s -m pip install scikit-build
python.exe -s -m pip install cmake
python.exe -s -m pip install llama-cpp-python
PAUSE