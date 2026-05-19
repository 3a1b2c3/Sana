@echo off
:: -----------------------------------------------------------------------------
:: SANA-WM environment installer for Windows.
::
:: Aligned with MIND's venv:  Python 3.10  +  torch 2.10.0+cu128.
:: (Upstream environment_setup_sana_wm.sh asks for 3.11 + torch 2.9.1; we
:: deviate so the same wheels work across MIND, scope-matrix2, and SANA-WM.)
::
:: torch is NOT installed by this script — install it into .venv-wm\ before
:: running, e.g.:
::   uv pip install --python .venv-wm\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
::
:: Creates .venv-wm\ next to this script and installs:
::   triton-windows (3.10+win replacement for upstream triton),
::   xformers latest-cu128, mmcv 1.7.2 (--no-build-isolation),
::   flash-linear-attention, liger_kernel, Pi3X (intrinsics),
::   plus everything else in requirements\sana_wm.txt minus the torch family
::   pins (so this script never touches torch/torchvision/torchaudio).
::
:: Caveat: flash-linear-attention's @triton.jit kernels may fail to import on
:: Python 3.10 with non-Windows triton because triton 3.5's source-inspection
:: regex assumes 3.11 semantics. triton-windows is patched separately; if
:: imports still blow up at "AttributeError: 'NoneType' object has no
:: attribute 'start'", bump to Python 3.11 by editing the `uv venv ... --python`
:: line below.
::
:: Usage:
::   environment_setup_sana_wm.bat                  build .venv-wm
::   environment_setup_sana_wm.bat D:\envs\sana-wm  custom venv path
::
:: Idempotent: rerunnable on an existing venv.
:: -----------------------------------------------------------------------------

setlocal enableextensions
cd /d "%~dp0"

set VENV=%~1
if not defined VENV set VENV=%~dp0.venv-wm

where uv >nul 2>&1
if errorlevel 1 ( echo ERROR: 'uv' not on PATH. Install from https://docs.astral.sh/uv/ & exit /b 2 )

echo [sana-wm] Target venv: %VENV%

if not exist "%VENV%\Scripts\python.exe" (
    echo [sana-wm] Creating Python 3.10.20 venv ^(matches MIND exactly^) ...
    uv venv "%VENV%" --python 3.10.20
    if errorlevel 1 exit /b %ERRORLEVEL%
)

set PY=%VENV%\Scripts\python.exe

echo [sana-wm] Verifying torch is pre-installed ^(this script does not install it^) ...
"%PY%" -c "import torch; print(' torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo ERROR: torch is not installed in %VENV%. Pre-install it first, e.g.:
    echo   uv pip install --python "%PY%" --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
    exit /b 2
)

echo [sana-wm] Base build tooling ^(pip, wheel, setuptools^<80 for mmcv^) ...
uv pip install --python "%PY%" -U pip wheel
if errorlevel 1 exit /b %ERRORLEVEL%
uv pip install --python "%PY%" "setuptools<80"
if errorlevel 1 exit /b %ERRORLEVEL%


echo [sana-wm] triton-windows ^(replaces upstream triton on Windows + Py 3.10^) ...
uv pip install --python "%PY%" "triton-windows>=3.4,<4"
if errorlevel 1 exit /b %ERRORLEVEL%

echo [sana-wm] xformers latest-cu128 ^(version solver matches torch 2.10^) ...
uv pip install --python "%PY%" --index-url https://download.pytorch.org/whl/cu128 xformers
if errorlevel 1 exit /b %ERRORLEVEL%

echo [sana-wm] mmcv 1.7.2 ^(--no-build-isolation; uses setuptools^<80 already in venv^) ...
uv pip install --python "%PY%" --no-build-isolation mmcv==1.7.2
if errorlevel 1 exit /b %ERRORLEVEL%

echo [sana-wm] Editable install of the repo ^(--no-deps; we resolve manually^) ...
uv pip install --python "%PY%" -e . --no-deps
if errorlevel 1 exit /b %ERRORLEVEL%

echo [sana-wm] Filtering requirements\sana_wm.txt to drop torch/triton/xformers pins ...
set FILTERED=%TEMP%\sana_wm_requirements_filtered.txt
powershell -NoProfile -Command "(Get-Content '%~dp0requirements\sana_wm.txt') | Where-Object { $_ -notmatch '^(torch|torchvision|torchaudio|triton|xformers)' } | Set-Content -Encoding utf8 '%FILTERED%'"
if not exist "%FILTERED%" ( echo ERROR: filtered requirements file not written: %FILTERED% & exit /b 1 )

:: Override `triton` to non-win32 only, so transitive triton deps from
:: flash-linear-attention/liger_kernel/xformers don't try to pull the
:: Linux-only `triton` distribution. triton-windows (installed above)
:: already provides `import triton` at runtime.
set OVERRIDES=%TEMP%\sana_wm_overrides.txt
> "%OVERRIDES%" echo triton; sys_platform != "win32"

echo [sana-wm] Installing filtered requirements ^(with triton override^) ...
uv pip install --python "%PY%" --override "%OVERRIDES%" -r "%FILTERED%"
if errorlevel 1 exit /b %ERRORLEVEL%

echo [sana-wm] Pi3X ^(--no-deps so torch/numpy aren't bumped^) ...
uv pip install --python "%PY%" --no-deps "git+https://github.com/yyfz/Pi3.git"
if errorlevel 1 exit /b %ERRORLEVEL%
uv pip install --python "%PY%" huggingface_hub opencv-python plyfile
if errorlevel 1 exit /b %ERRORLEVEL%

echo [sana-wm] flash-attn ^(optional source build; needs vcvarsall.bat x64 + cu128 nvcc^) ...
uv pip install --python "%PY%" --no-build-isolation "flash-attn>=2.7.0"
if errorlevel 1 ( echo WARNING: flash-attn build failed. SANA-WM falls back to xformers/SDPA; continuing. )

echo.
echo [sana-wm] Done. Inference python: %PY%
echo [sana-wm] Next: test_sana_wm.bat
exit /b 0
