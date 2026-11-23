@echo off
rem One-click launch of the demo in Docker. Uses the NVIDIA GPU when one is available.
setlocal
cd /d "%~dp0"
title Planogram compliance demo (Docker)

docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running. Start Docker Desktop and run this file again.
    pause
    exit /b 1
)

set "FILES=-f docker-compose.yml"
nvidia-smi >nul 2>&1 && set "FILES=-f docker-compose.yml -f docker-compose.gpu.yml"
if "%FILES%"=="-f docker-compose.yml" (echo No NVIDIA GPU found, running on CPU.) else (echo NVIDIA GPU found, running on GPU.)

echo Building the image (first time: about 10 GB, several minutes) and starting the container ...
docker compose %FILES% up --build -d
if errorlevel 1 (
    echo ERROR: docker compose failed. See the messages above.
    pause
    exit /b 1
)

echo Waiting for the demo to answer on http://localhost:7860 ...
for /l %%i in (1,1,60) do (
    curl -s -o nul http://localhost:7860 && goto up
    timeout /t 2 >nul
)
echo The container started but the page is not answering yet. Check: docker compose logs -f
pause
exit /b 1

:up
start "" http://localhost:7860
echo Demo is running. Stop it with: docker compose down
pause
