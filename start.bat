@echo off
cd /d "%~dp0"
echo Starting D&D transcription...
.venv\Scripts\python.exe transcribe.py %*
if errorlevel 1 (
    echo.
    echo Script exited with an error. Press any key to close.
    pause > nul
)
