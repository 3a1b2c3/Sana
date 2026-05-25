@echo off
:: Launch the SANA-WM Gradio app (app\app_sana_wm.py) for interactive single-image use.
::
:: For batch CLI runs over the three bundled demos, see test_sana_wm.bat instead.
::
:: Prereqs:
::   1. .venv-wm built via environment_setup_sana_wm.bat.
::   2. SANA-WM_bidirectional weights resolved via HF Hub cache. The first run
::      pulls ~94 GB into %USERPROFILE%\.cache\huggingface\hub\. Subsequent
::      runs are instant. To pre-fetch:
::          python tools\download_sana_wm.py
::      For a local copy elsewhere: tools\download_sana_wm.py --dest <path>
::
:: Usage:
::   gradio_sana_wm.bat                          UI on default port 7860
::   gradio_sana_wm.bat --no_refiner             skip the LTX-2 refiner (faster)
::   gradio_sana_wm.bat --offload_vae --offload_refiner    tight-VRAM mode
::   gradio_sana_wm.bat --share                  expose a public gradio.live URL
::   gradio_sana_wm.bat --server_port 8000       custom port

setlocal enableextensions
cd /d "%~dp0"

:: Pin triton's CUDA toolkit to v12.8 (matches torch 2.8.0+cu128).
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"

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

set ENTRY=%~dp0app\app_sana_wm.py
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set TOKENIZERS_PARALLELISM=false

if not exist "%ENTRY%"       ( echo ERROR: Gradio app not found: %ENTRY% & exit /b 2 )

:: Resolve MODEL_ROOT from the HF Hub cache. snapshot_download is a no-op if
:: all files are already cached; otherwise it downloads the missing pieces
:: into ~/.cache/huggingface/hub/ (not into output/pretrained_models/).
for /f "delims=" %%P in ('python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='Efficient-Large-Model/SANA-WM_bidirectional'))"') do set MODEL_ROOT=%%P
if not defined MODEL_ROOT     ( echo ERROR: Failed to resolve SANA-WM via HF Hub cache & exit /b 2 )
if not exist "%MODEL_ROOT%"   ( echo ERROR: HF cache path does not exist: %MODEL_ROOT% & exit /b 2 )
echo MODEL_ROOT=%MODEL_ROOT%

echo ============================================================
echo SANA-WM Gradio  ^|  open http://127.0.0.1:7860 once UI launches
echo ============================================================

python "%ENTRY%" --config "%MODEL_ROOT%\config.yaml" --model_path "%MODEL_ROOT%\dit\sana_wm_1600m_720p.safetensors" --refiner_checkpoint "%MODEL_ROOT%\refiner\refiner.safetensors" --refiner_gemma_root "%MODEL_ROOT%\refiner\text_encoder" %*

exit /b %ERRORLEVEL%
