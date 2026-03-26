@echo off
title Kenya Health Gap — API
cd /d "C:\Projects\kenya-health-gap"

echo Activating environment...
call venv\Scripts\activate

echo.
echo  ================================
echo   Kenya Health Gap — FastAPI
echo  ================================
echo   API:  http://localhost:8000
echo   Docs: http://localhost:8000/docs
echo  ================================
echo.
echo  Keep this window open.
echo  Start the dashboard in a second window.
echo.

uvicorn main:app --reload --port 8000
pause
