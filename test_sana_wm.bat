@echo off
:: SANA-WM smoke test on asset\sana_wm\demo_0 (image + prompt + action DSL).
::
:: Prereqs:
::   1. PR #379 merged into the tree (you have it — inference_sana_wm.py + asset\sana_wm\ present).
::   2. Python 3.11 venv at .venv-wm\ built via environment_setup_sana_wm.bat.
::   3. SANA-WM_bidirectional weights downloaded under
::      output\pretrained_models\SANA-WM_bidirectional\.
::
:: Usage:
::   test_sana_wm.bat                                              defaults from run_sana_wm.sh
::   test_sana_wm.bat --num_frames 81 --no_refiner --offload_vae   smaller/faster smoke
::
:: Any extra args pass through to inference_sana_wm.py (argparse, no pyrallis).

setlocal enableextensions
cd /d "%~dp0"

set VENV=%~dp0.venv-wm
set PY=%VENV%\Scripts\python.exe
set ENTRY=%~dp0inference_video_scripts\inference_sana_wm.py
set MODEL_ROOT=%~dp0output\pretrained_models\SANA-WM_bidirectional
set IMAGE=%~dp0asset\sana_wm\demo_0.png
set PROMPT=%~dp0asset\sana_wm\demo_0.txt
set OUT_DIR=%~dp0results\sana_wm_demo

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set TOKENIZERS_PARALLELISM=false

if not exist "%PY%"          ( echo ERROR: 3.11 venv python not found: %PY% & echo Run environment_setup_sana_wm.bat first. & exit /b 2 )
if not exist "%ENTRY%"       ( echo ERROR: inference entry not found: %ENTRY% & exit /b 2 )
if not exist "%MODEL_ROOT%"  ( echo ERROR: SANA-WM model dir not found: %MODEL_ROOT% & exit /b 2 )
if not exist "%IMAGE%"       ( echo ERROR: demo image not found: %IMAGE% & exit /b 2 )
if not exist "%PROMPT%"      ( echo ERROR: demo prompt not found: %PROMPT% & exit /b 2 )
if not exist "%MODEL_ROOT%\dit\sana_wm_1600m_720p.safetensors" ( echo ERROR: DiT weights missing under %MODEL_ROOT%\dit\ & exit /b 2 )
if not exist "%MODEL_ROOT%\refiner\refiner.safetensors" ( echo ERROR: refiner weights missing under %MODEL_ROOT%\refiner\ & exit /b 2 )

echo ============================================================
echo SANA-WM smoke test  (demo_0)
echo ============================================================
echo   venv    : %VENV%
echo   model   : %MODEL_ROOT%
echo   image   : %IMAGE%
echo   prompt  : %PROMPT%
echo   out     : %OUT_DIR%
echo ============================================================

"%PY%" "%ENTRY%" --image "%IMAGE%" --prompt "%PROMPT%" --action "w-80,jw-40,w-40,lw-60,w-100" --translation_speed 0.055 --rotation_speed_deg 1.2 --output_dir "%OUT_DIR%" --name demo_0 --num_frames 321 --fps 16 --step 60 --cfg_scale 5.0 --flow_shift 8.0 --config "%MODEL_ROOT%\config.yaml" --model_path "%MODEL_ROOT%\dit\sana_wm_1600m_720p.safetensors" --refiner_checkpoint "%MODEL_ROOT%\refiner\refiner.safetensors" --refiner_gemma_root "%MODEL_ROOT%\refiner\text_encoder" %*

exit /b %ERRORLEVEL%
