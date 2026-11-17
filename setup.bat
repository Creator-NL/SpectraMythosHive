@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Spectra Mythos Hive - Windows Environment Setup
echo ============================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10+ ^(64-bit^) and check "Add Python to PATH".
    pause
    exit /b 1
)

if not exist "venv\" (
    echo [1/4] Creating Python virtual environment ^(venv^)...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b %errorlevel%
    )
) else (
    echo [1/4] Existing venv found. Reusing it...
)

echo [2/4] Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip

echo.
echo [3/4] Select PyTorch backend for this PC:
echo   1 = NVIDIA RTX 20 / 30 / 40 series ^(e.g. RTX 4060 8GB^) - CUDA 12.6
echo   2 = NVIDIA RTX 50 series / Blackwell ^(e.g. RTX 5060 Ti 16GB^) - CUDA 12.8
echo   3 = CPU only ^(ideal if this PC only runs hive_server.py^)
echo.
set /p GPU_CHOICE="Enter choice [1/2/3] (default 1): "
if "%GPU_CHOICE%"=="" set GPU_CHOICE=1

if "%GPU_CHOICE%"=="1" (
    echo Installing PyTorch CUDA 12.6...
    venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
) else if "%GPU_CHOICE%"=="2" (
    echo Installing PyTorch CUDA 12.8...
    venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
) else if "%GPU_CHOICE%"=="3" (
    echo Installing CPU-only PyTorch...
    venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
) else (
    echo Invalid choice "%GPU_CHOICE%". Aborting.
    pause
    exit /b 1
)

if %errorlevel% neq 0 (
    echo [ERROR] PyTorch installation failed.
    pause
    exit /b %errorlevel%
)

echo.
echo [4/4] Installing Hive ^& OpenMythos dependencies from requirements.txt...
venv\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b %errorlevel%
)

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo.
    echo Created .env from .env.example.
    echo Remember to edit .env and set your HIVE_PASSWORD / HIVE_URL / WORKER_NAME!
)

echo.
echo ============================================================
echo   SETUP COMPLETE!
echo   - Host PC  : Run StartHive.bat
echo   - Worker PC: Run "StartWorker - phase1.bat"
echo   - Test AI  : Run "StartModelTester - phase1.bat"
echo ============================================================
pause
