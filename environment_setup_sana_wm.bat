@echo off
:: -----------------------------------------------------------------------------
:: SANA-WM environment installer for Windows (port of environment_setup_sana_wm.sh).
::
:: Creates a Python 3.11 venv at .venv-wm/ and installs the SANA-WM stack:
::   torch 2.9.1+cu128, triton 3.5.1 (or triton-windows fallback), xformers
::   0.0.33.post2, transformers 4.57.3, mmcv 1.7.2 (no-build-isolation),
::   flash-linear-attention, liger_kernel, Pi3X, plus everything in
::   requirements\sana_wm.txt.
::
:: We skip the conda + cuda-toolkit steps from the .sh: on Windows the cu128
:: torch wheels ship their own CUDA libs, and source builds (mmcv, flash-attn)
:: use MSVC + the cu128 toolchain you set up via vcvarsall.bat x64.
::
:: Usage:
::   environment_setup_sana_wm.bat                  create .venv-wm at repo root
::   environment_setup_sana_wm.bat D:\envs\sana-wm  custom venv path
::
:: Idempotent: rerunning on an existing venv reconciles versions instead of failing.
:: -----------------------------------------------------------------------------

setlocal enableextensions
cd /d "%~dp0"

set VENV=%~1
if not defined VENV set VENV=%~dp0.venv-wm

where uv >nul 2>&1
if errorlevel 1 ( echo ERROR: 'uv' not on PATH. Install from https://docs.astral.sh/uv/ & exit /b 2 )

echo [sana-wm] Target venv: %VENV%

if not exist "%VENV%\Scripts\python.exe" (
    echo [sana-wm] Creating venv with Python 3.11 ...
    uv venv "%VENV%" --python 3.11
    if errorlevel 1 exit /b %ERRORLEVEL%
)

set PY=%VENV%\Scripts\python.exe

echo [sana-wm] Base build tooling ^(pip, wheel, setuptools^<80^) ...
uv pip install --python "%PY%" -U pip wheel
if errorlevel 1 exit /b %ERRORLEVEL%
uv pip install --python "%PY%" "setuptools<80"
if errorlevel 1 exit /b %ERRORLEVEL%

echo [sana-wm] Torch 2.9.1 + torchvision 0.24.1 + torchaudio 2.9.1 ^(cu128^) ...
uv pip install --python "%PY%" --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
if errorlevel 1 exit /b %ERRORLEVEL%

echo [sana-wm] triton ^(prefer 3.5.1, fall back to triton-windows^) ...
uv pip install --python "%PY%" triton==3.5.1
if errorlevel 1 ( echo [sana-wm]   triton 3.5.1 unavailable, trying triton-windows ... & uv pip install --python "%PY%" "triton-windows>=3.5,<3.6" )
if errorlevel 1 exit /b %ERRORLEVEL%

echo [sana-wm] xformers 0.0.33.post2 ^(cu128^) ...
uv pip install --python "%PY%" --index-url https://download.pytorch.org/whl/cu128 xformers==0.0.33.post2
if errorlevel 1 exit /b %ERRORLEVEL%

echo [sana-wm] mmcv 1.7.2 ^(--no-build-isolation; needs setuptools^<80 in the venv^) ...
uv pip install --python "%PY%" --no-build-isolation mmcv==1.7.2
if errorlevel 1 exit /b %ERRORLEVEL%

echo [sana-wm] Editable install of the repo ^(--no-deps^) ...
uv pip install --python "%PY%" -e . --no-deps
if errorlevel 1 exit /b %ERRORLEVEL%

echo [sana-wm] requirements\sana_wm.txt ...
uv pip install --python "%PY%" -r requirements\sana_wm.txt
if errorlevel 1 exit /b %ERRORLEVEL%

echo [sana-wm] Pi3X ^(--no-deps so torch/numpy aren't overwritten^) ...
uv pip install --python "%PY%" --no-deps "git+https://github.com/yyfz/Pi3.git"
if errorlevel 1 exit /b %ERRORLEVEL%
uv pip install --python "%PY%" huggingface_hub opencv-python plyfile
if errorlevel 1 exit /b %ERRORLEVEL%

echo [sana-wm] flash-attn ^(source build, 20-40 min; needs vcvarsall.bat x64 + cu128 nvcc on PATH^) ...
uv pip install --python "%PY%" --no-build-isolation "flash-attn>=2.7.0"
if errorlevel 1 ( echo WARNING: flash-attn build failed. SANA-WM falls back to xformers/SDPA; continuing. )

echo.
echo [sana-wm] Done. Inference python: %PY%
echo [sana-wm] Next: run test_sana_wm.bat
exit /b 0
