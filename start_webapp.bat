@echo off
title MRI Enhancement Demo Server
cd /d "C:\Projects\Yugma"
echo Starting the MRI Enhancement demo website...
echo It will open in your browser at http://localhost:5000
echo Keep this window open during the demo. Close it (or press Ctrl+C) to stop.
echo.
rem open the browser a few seconds after the server starts
start "" cmd /c "timeout /t 4 >nul & start http://localhost:5000"
"C:\Users\RAFAN AHAMAD SHEIK\.conda\envs\tfenv\python.exe" webapp.py
pause
