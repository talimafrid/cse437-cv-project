@echo off
rem One-click setup and launch of the planogram compliance demo (Windows).
rem First run: creates .venv and installs packages (about 5 GB download). Later runs start in seconds.
setlocal
cd /d "%~dp0"
title Planogram compliance demo

rem ---- 1. Python 3.13 virtual environment ------------------------------------------------
if exist ".venv\Scripts\python.exe" goto venv_ok
echo [1/4] Creating virtual environment in .venv ...
py -3.13 -m venv .venv 2>nul || python -m venv .venv
if errorlevel 1 (
    echo ERROR: Python 3.13 not found. Install it from https://www.python.org/downloads/ and tick "Add to PATH".
    goto fail
)
:venv_ok
set "PY=.venv\Scripts\python.exe"

rem ---- 2. Packages (only when requirements.txt changed since the last install) ------------
fc /b requirements.txt ".venv\installed-requirements.txt" >nul 2>&1
if not errorlevel 1 goto deps_ok
echo [2/4] Installing packages (first run takes a while) ...
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo ERROR: package install failed. Check your internet connection and run this file again.
    goto fail
)
copy /y requirements.txt ".venv\installed-requirements.txt" >nul
:deps_ok

rem ---- 3. Model, gallery index and shelf images --------------------------------------------
echo [3/4] Checking model and data ...
if not exist "models\sku110k_yolo11s.pt" (
    echo ERROR: models\sku110k_yolo11s.pt is missing. Copy the trained detector there, see README.md "What you need".
    goto fail
)
if not exist "data\grozi\Testing\store1\images" (
    echo ERROR: GroZi-3.2k shelf images not found in data\grozi\Testing. See README.md "What you need".
    goto fail
)
if exist "outputs\index\siglip2_b_v8\emb.npy" goto index_ok
if not exist "data\grozi\Training\Food" (
    echo ERROR: no gallery index in outputs\index\siglip2_b_v8 and no studio images in data\grozi\Training to build one.
    goto fail
)
echo Building the product gallery index (about 10 min on a GPU, much longer on CPU) ...
"%PY%" -m src.gallery build --backbone siglip2_b
if errorlevel 1 goto fail
:index_ok

rem ---- 4. Launch -----------------------------------------------------------------------------
echo [4/4] Starting the demo at http://127.0.0.1:7860 (your browser opens by itself). Close this window to stop it.
set OPEN_BROWSER=1
"%PY%" -m src.app
if errorlevel 1 goto fail
exit /b 0

:fail
echo.
echo Setup stopped. Fix the error above and double-click run_demo.bat again.
pause
exit /b 1
