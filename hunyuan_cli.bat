@echo off
REM Automatic1112 — Headless CLI Launcher (Windows)
setlocal

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

if defined A1112_VENV_PATH (
    if exist "%A1112_VENV_PATH%\Scripts\activate.bat" (
        call "%A1112_VENV_PATH%\Scripts\activate.bat"
        goto :venv_done
    )
)
for %%D in (venv .venv hunyuan_env) do (
    if exist "%PROJECT_ROOT%\%%D\Scripts\activate.bat" (
        call "%PROJECT_ROOT%\%%D\Scripts\activate.bat"
        goto :venv_done
    )
)
:venv_done

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set "HF_HOME=%PROJECT_ROOT%.hf_cache"
set "TORCH_HOME=%PROJECT_ROOT%.torch_cache"
if not defined HF_HUB_OFFLINE set HF_HUB_OFFLINE=0

python -m hunyuan_desktop.cli %*

endlocal
