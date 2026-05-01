@echo off
cd /d "%~dp0"
echo Starting DnD transcription v2 (Whisper large-v3 + diarization)...
.venv_v2\Scripts\python.exe transcribe_v2.py %*
if errorlevel 1 (
    echo.
    echo Script exited with an error. Press any key to close.
    pause > nul
)
