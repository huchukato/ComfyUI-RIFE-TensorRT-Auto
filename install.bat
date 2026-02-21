@echo off
REM Auto-install script for ComfyUI RIFE TensorRT (Windows)
REM Detects CUDA version and installs appropriate requirements

echo 🔍 Detecting CUDA version...

REM Try to detect CUDA version
set CUDA_VERSION=
where nvcc >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=6 delims=, " %%i in ('nvcc --version ^| findstr "release"') do set CUDA_VERSION=%%i
    echo ✅ Found CUDA version: %CUDA_VERSION%
) else if defined CUDA_PATH (
    if exist "%CUDA_PATH%\bin\nvcc.exe" (
        for /f "tokens=6 delims=, " %%i in ('"%CUDA_PATH%\bin\nvcc.exe" --version ^| findstr "release"') do set CUDA_VERSION=%%i
        echo ✅ Found CUDA version via CUDA_PATH: %CUDA_VERSION%
    )
) else if defined CUDA_HOME (
    if exist "%CUDA_HOME%\bin\nvcc.exe" (
        for /f "tokens=6 delims=, " %%i in ('"%CUDA_HOME%\bin\nvcc.exe" --version ^| findstr "release"') do set CUDA_VERSION=%%i
        echo ✅ Found CUDA version via CUDA_HOME: %CUDA_VERSION%
    )
)

if "%CUDA_VERSION%"=="" (
    echo ⚠️  Could not detect CUDA version automatically
    echo Please ensure CUDA is installed and nvcc is in your PATH
    echo Or set CUDA_PATH or CUDA_HOME environment variables
    pause
    exit /b 1
)

REM Extract major version
for /f "tokens=1 delims=." %%i in ("%CUDA_VERSION%") do set CUDA_MAJOR=%%i

echo 📦 Installing requirements for CUDA %CUDA_MAJOR%...

REM Install appropriate requirements based on CUDA version
if "%CUDA_MAJOR%"=="13" (
    echo 🚀 Using CUDA 13 requirements ^(RTX 50 series^)
    pip install -r requirements.txt
) else if "%CUDA_MAJOR%"=="12" (
    echo 🔧 Using CUDA 12 requirements ^(RTX 30/40 series^)
    pip install -r requirements_cu12.txt
) else (
    echo ❌ Unsupported CUDA version: %CUDA_VERSION%
    echo Supported versions: CUDA 12.x, CUDA 13.x
    pause
    exit /b 1
)

if %ERRORLEVEL% EQU 0 (
    echo ✅ Installation completed successfully!
    echo 🎯 You can now use the ComfyUI RIFE TensorRT node
) else (
    echo ❌ Installation failed!
    pause
    exit /b 1
)

pause
