@echo off
chcp 65001 > nul
SETLOCAL ENABLEDELAYEDEXPANSION
echo 🛠️ Установка и запуск веб-сервера Whisper
echo ==========================================

REM Активируем виртуальное окружение
call .venv\Scripts\activate

REM Запускаем скрипт установки
python setup-web-server.py

pause
