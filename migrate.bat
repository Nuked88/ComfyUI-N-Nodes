@echo off
setlocal

rem Check if the file path is provided
if "%1"=="" (
  echo Error: provide the path of the text file to migrate.
  exit /b 1
)

rem Check if the file exists
if not exist "%1" (
  echo Error: the file %1 does not exist.
  exit /b 1
)

rem Run the Python script to migrate the file
python %~dp0libs\migrate.py "%~f1"

echo Replacement completed successfully.

pause
