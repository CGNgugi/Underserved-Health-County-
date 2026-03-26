@echo off
title Kenya Health Gap — Dashboard
cd /d "C:\Projects\kenya-health-gap"

echo Activating environment...
call venv\Scripts\activate

echo.
echo  ================================
echo   Kenya Health Gap — Dashboard
echo  ================================
echo   Dashboard: http://localhost:8501
echo  ================================
echo.
echo  Make sure 1_start_api.bat is running first.
echo.

streamlit run ucs.py
pause
