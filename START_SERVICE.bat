@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Forte - запуск для жюри
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_all.ps1"
echo.
pause
