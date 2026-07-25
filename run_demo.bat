@echo off
title MedhaDrishti - MRI Enhancement Demo
cd /d "%~dp0"
echo.
echo   MRI Enhancement ^& ROI Segmentation - live demo
echo   ---------------------------------------------
echo   Starting server... your browser will open at http://localhost:5000
echo   Keep this window open during the demo. Press Ctrl+C to stop.
echo.
start "" cmd /c "timeout /t 5 >nul & start http://localhost:5000"
"C:\Users\RAFAN AHAMAD SHEIK\.conda\envs\tfenv\python.exe" src\webapp.py
pause
