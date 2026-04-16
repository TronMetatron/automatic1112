@echo off
REM Automatic1112 — Desktop Application Launcher (Windows)

setlocal

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

echo === Automatic1112 Desktop ===
echo Project: %PROJECT_ROOT%

REM Virtual environment — auto-detect or use A1112_VENV_PATH
if defined A1112_VENV_PATH (
    if exist "%A1112_VENV_PATH%\Scripts\activate.bat" (
        echo Activating venv: %A1112_VENV_PATH%
        call "%A1112_VENV_PATH%\Scripts\activate.bat"
        goto :venv_done
    )
)

REM Auto-detect common locations
for %%D in (venv .venv hunyuan_env) do (
    if exist "%PROJECT_ROOT%\%%D\Scripts\activate.bat" (
        echo Activating venv: %PROJECT_ROOT%\%%D
        call "%PROJECT_ROOT%\%%D\Scripts\activate.bat"
        goto :venv_done
    )
)

echo No virtual environment found. Using system Python.
echo   Tip: set A1112_VENV_PATH or create a venv in %PROJECT_ROOT%\venv

:venv_done

REM CUDA memory config
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Cache directories
set "HF_HOME=%PROJECT_ROOT%.hf_cache"
set "TORCH_HOME=%PROJECT_ROOT%.torch_cache"

if not defined HF_HUB_OFFLINE set HF_HUB_OFFLINE=0

echo Launching desktop application...
echo.

python -m hunyuan_desktop.main %*

endlocal
