@echo off
setlocal

:: 获取当前脚本所在目录
set "DIR=%~dp0"

echo ========================================================
echo        Scrapling 自适应爬虫图形控制台 - 一键启动
echo ========================================================

:: 检查 Python 环境
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请确保已安装 Python 并在安装时勾选 "Add Python to PATH"
    pause
    goto :EOF
)

:: 创建虚拟环境 (如果不存在)
if not exist "%DIR%venv" (
    echo [状态] 正在创建虚拟环境 ^(首次启动可能需要一些时间^)...
    python -c "import venv; venv.create('%DIR%venv', with_pip=True)"
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败
        pause
        goto :EOF
    )
)

:: 激活虚拟环境
echo [状态] 激活虚拟环境...
call "%DIR%venv\Scripts\activate.bat"

:: 安装依赖
echo [状态] 检查并安装相关依赖...
pip install -r "%DIR%requirements.txt"

:: 启动应用程序
echo [状态] 启动应用程序...
python "%DIR%main.py"

endlocal
pause
