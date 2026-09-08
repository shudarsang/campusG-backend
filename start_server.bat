@echo off
REM Starts the CampusGuide AI backend (FastAPI + uvicorn).
REM Run this from anywhere; it switches to its own folder first.

cd /d "%~dp0"

call ..\.venv\Scripts\activate.bat

uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

pause
