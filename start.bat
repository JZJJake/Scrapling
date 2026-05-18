@echo off
chcp 65001 >nul
setlocal

:: Get current script directory
set "DIR=%~dp0"

echo ========================================================
echo        Scrapling App - One-Click Start
echo ========================================================

:: Check Python environment
python --version >nul 2>&1
if errorlevel 1 (
    echo [Error] Python not found. Please install Python and check "Add Python to PATH".
    pause
    goto :EOF
)

:: Create virtual environment (if not exists)
if not exist "%DIR%venv" (
    echo [Status] Creating virtual environment...
    python -m venv "%DIR%venv"
    if errorlevel 1 (
        echo [Error] Failed to create virtual environment.
        pause
        goto :EOF
    )
)

:: Activate virtual environment
echo [Status] Activating virtual environment...
call "%DIR%venv\Scripts\activate.bat"

:: Install dependencies
echo [Status] Installing dependencies...
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r "%DIR%requirements.txt"

:: Install Playwright Chromium dependencies (for DynamicFetcher)
echo [Status] Installing Playwright Chromium dependencies...
playwright install chromium

:: Configure HuggingFace mirror for domestic downloads
set HF_ENDPOINT=https://hf-mirror.com

:: Run application
echo [Status] Starting ScraplingApp...
python "%DIR%main.py"

endlocal
pause
