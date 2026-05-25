@echo off
:: SANA-WM CLI smoke test — runs all three bundled demos (demo_0, demo_1, demo_2)
:: through inference_sana_wm.py. Saves one mp4 per demo to results\sana_wm_demo\.
::
:: For interactive single-image use (browser UI), see gradio_sana_wm.bat.
::
:: Prereqs:
::   1. .venv-wm built via environment_setup_sana_wm.bat.
::   2. SANA-WM_bidirectional weights at
::      output\pretrained_models\SANA-WM_bidirectional\ with the local-path
::      vae_pretrained fix in config.yaml so HF doesn't redownload.
::   3. PR #379 in the tree (inference_sana_wm.py + asset\sana_wm\ present).
::
:: Usage:
::   test_sana_wm.bat                          all three demos, defaults
::   test_sana_wm.bat --num_frames 81 --no_refiner --offload_vae   cheap smoke
::   test_sana_wm.bat --demo demo_1            single demo only
::
:: Extra args after --demo X pass through to inference_sana_wm.py.

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

:: Pin triton's CUDA toolkit to v12.8 (matches torch 2.8.0+cu128). With v13.0
:: also installed, triton otherwise grabs v13.0 from CUDA_PATH/nvcc-on-PATH and
:: fails to link its JIT kernels against the cu128-built torch wheel.
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"

:: Strip host-venv pollution + activate .venv-wm so `python` resolves correctly.
set "VIRTUAL_ENV="
set "VIRTUAL_ENV_PROMPT="
set "PYTHONHOME="
set "PYTHONPATH="
set "UV_PYTHON="
set "UV_PROJECT_ENVIRONMENT="
if exist "%~dp0.venv-wm\Scripts\activate.bat" (
    call "%~dp0.venv-wm\Scripts\activate.bat"
) else (
    echo ERROR: .venv-wm not found. Run environment_setup_sana_wm.bat first.
    exit /b 2
)

set ENTRY=%~dp0inference_video_scripts\inference_sana_wm.py
set MODEL_ROOT=%~dp0output\pretrained_models\SANA-WM_bidirectional
set ASSET=%~dp0asset\sana_wm
set OUT_DIR=%~dp0results\sana_wm_demo
set ACTION=w-80,jw-40,w-40,lw-60,w-100

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set TOKENIZERS_PARALLELISM=false
:: triton-windows lacks `triton.compiler.compiler.triton_key` so PyTorch 2.8's
:: inductor backend explodes when @torch.compile fires on Sana's GDN gates.
:: Disable dynamo entirely; eager fallback is slower but works.
set TORCHDYNAMO_DISABLE=1
:: Stay offline once tokenizers/models are cached — no per-run HF hub roundtrip.
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

:: --demo <name> limits to a single demo. Anything else is passed through.
set DEMOS=demo_0 demo_1 demo_2
set PASSTHROUGH=
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--demo" ( set DEMOS=%~2 & shift & shift & goto parse_args )
set PASSTHROUGH=%PASSTHROUGH% %1
shift
goto parse_args
:args_done

if not exist "%ENTRY%"       ( echo ERROR: inference entry not found: %ENTRY% & exit /b 2 )
if not exist "%MODEL_ROOT%"  ( echo ERROR: SANA-WM model dir not found: %MODEL_ROOT% & exit /b 2 )
if not exist "%MODEL_ROOT%\dit\sana_wm_1600m_720p.safetensors" ( echo ERROR: DiT weights missing & exit /b 2 )
:: Refiner pre-check skipped: the python call below passes --no_refiner, so
:: the single-file refiner.safetensors path isn't actually loaded. What's on
:: disk is the HF sharded layout (refiner/connectors/ + refiner/text_encoder/),
:: which the inference script ignores under --no_refiner. Re-enable this
:: check if you ever drop --no_refiner from the python command.

echo ============================================================
echo SANA-WM CLI smoke test  ^| demos: %DEMOS%
echo ============================================================
echo   model   : %MODEL_ROOT%
echo   assets  : %ASSET%
echo   out     : %OUT_DIR%
echo   action  : %ACTION%
echo   extras  :%PASSTHROUGH%
echo ============================================================

for %%D in (%DEMOS%) do (
    set "IMAGE=%ASSET%\%%D.png"
    set "PROMPT=%ASSET%\%%D.txt"
    set "INTRINSICS=%ASSET%\%%D_intrinsics.npy"
    if not exist "!IMAGE!"      ( echo ERROR: missing !IMAGE!     & exit /b 2 )
    if not exist "!PROMPT!"     ( echo ERROR: missing !PROMPT!    & exit /b 2 )
    if not exist "!INTRINSICS!" ( echo ERROR: missing !INTRINSICS! & exit /b 2 )

    echo.
    echo --- %%D ---
    python "%ENTRY%" --image "!IMAGE!" --prompt "!PROMPT!" --action "%ACTION%" --translation_speed 0.055 --rotation_speed_deg 1.2 --intrinsics "!INTRINSICS!" --output_dir "%OUT_DIR%" --name %%D --num_frames 81 --fps 16 --step 20 --cfg_scale 5.0 --flow_shift 8.0 --no_refiner --config "%MODEL_ROOT%\config.yaml" --model_path "%MODEL_ROOT%\dit\sana_wm_1600m_720p.safetensors" --refiner_checkpoint "%MODEL_ROOT%\refiner\refiner.safetensors" --refiner_gemma_root "%MODEL_ROOT%\refiner\text_encoder"%PASSTHROUGH%
    if errorlevel 1 ( echo FAIL on %%D ^(rc=!ERRORLEVEL!^) & exit /b !ERRORLEVEL! )
)

echo.
echo All demos done. Output under %OUT_DIR%
exit /b 0
