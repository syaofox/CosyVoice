@echo off
set PATH=%PATH%;%MY_PATH%
set GRADIO_TEMP_DIR=%~dp0TEMP

call conda activate cosyvoice
python .\role_tts.py
pause