@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem Перезапуск ТОЛЬКО сервиса (окно «Forte · сервис»). Туннель и публичный адрес не трогаются.
start "Forte - сервис" powershell -NoProfile -ExecutionPolicy Bypass -NoExit -File "%~dp0tools\run_service.ps1"
