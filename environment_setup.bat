@echo off
:: Sana environment setup for Windows.
::
:: Companion to environment_setup.sh (Linux/macOS). Applies two Windows-only
:: install constraints that aren't expressible in pyproject.toml without
:: affecting Linux:
::   1. mmcv==1.7.2's setup.py imports `pkg_resources`, removed in setuptools 81.
::      We pass `--build-constraint setuptools_lt70.txt` so uv's isolated build
::      env picks an older setuptools that still ships pkg_resources.
::   2. mmcv's CUDA ops can't compile on Windows without configured MSVC+CUDA.
::      MMCV_WITH_OPS=0 builds only the pure-Python portion (Sana only uses
::      mmcv.Registry / mmcv.runner / mmcv.utils at inference time).
::
:: Triton on Windows is handled by `; sys_platform == 'win32'` markers in
:: pyproject.toml (triton-windows wheel replaces the Linux triton wheel).
::
:: Usage:
::   environment_setup.bat                    uses .\.venv
::   environment_setup.bat C:\path\to\.venv   override venv location

setlocal enableextensions
cd /d "%~dp0"

set VENV=%~1
if "%VENV%"=="" set VENV=%~dp0.venv

set PY=%VENV%\Scripts\python.exe
if not exist "%PY%" (
    echo Creating venv at %VENV% with Python 3.10 ...
    uv venv --python 3.10 "%VENV%"
    if errorlevel 1 ( echo ERROR: uv venv failed & exit /b 1 )
)

set CONSTRAINT=%~dp0setuptools_lt70.txt
> "%CONSTRAINT%" echo setuptools^<70

echo.
echo Installing Sana into %VENV% ...
echo   (torch 2.8.0+cu128 + xformers + triton-windows + mmcv 1.7.2)
echo.
set MMCV_WITH_OPS=0
uv pip install --python "%PY%" --build-constraint "%CONSTRAINT%" -e "%~dp0"
if errorlevel 1 (
    echo.
    echo ERROR: uv pip install failed.
    exit /b 1
)

echo.
echo Verifying critical imports ...
"%PY%" -c "import triton, imageio, mmcv, mmcv.runner, transformers, diffusers, sana; print('Sana env OK: triton', triton.__version__, '| mmcv', mmcv.__version__)"
if errorlevel 1 (
    echo ERROR: import verification failed.
    exit /b 1
)

echo.
echo Done.
echo Note: flash-attn install is NOT automated on Windows. If needed, use a prebuilt wheel
echo (e.g. https://github.com/mjun0812/flash-attention-prebuild-wheels) matching
echo torch 2.8.0+cu128 / Python 3.10.
endlocal
