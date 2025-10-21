@echo off
REM SmartRAG Docker Quick Start Script for Windows

echo ========================================
echo SmartRAG Docker Deployment
echo ========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not in PATH
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo Docker is installed. Checking Docker Compose...
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker Compose is not installed
    echo Please install Docker Compose
    pause
    exit /b 1
)

echo.
echo Select deployment option:
echo 1. Full Stack (App + Ollama in Docker) - Recommended
echo 2. Lightweight (App in Docker, Ollama on host)
echo.
set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo Starting Full Stack deployment...
    echo This will take 10-15 minutes on first run (downloading models)
    echo.
    docker-compose up -d
    if %errorlevel% equ 0 (
        echo.
        echo ========================================
        echo SmartRAG is starting!
        echo ========================================
        echo.
        echo Access the application at: http://localhost:8501
        echo.
        echo To view logs: docker-compose logs -f
        echo To stop: docker-compose down
        echo.
    ) else (
        echo.
        echo ERROR: Failed to start SmartRAG
        echo Check the logs with: docker-compose logs
    )
) else if "%choice%"=="2" (
    echo.
    echo Starting Lightweight deployment...
    echo.
    echo IMPORTANT: Make sure Ollama is running on your host machine!
    echo If not running, open another terminal and run: ollama serve
    echo.
    pause
    docker-compose -f docker-compose.lite.yml up -d
    if %errorlevel% equ 0 (
        echo.
        echo ========================================
        echo SmartRAG Lightweight is running!
        echo ========================================
        echo.
        echo Access the application at: http://localhost:8501
        echo.
        echo To view logs: docker-compose -f docker-compose.lite.yml logs -f
        echo To stop: docker-compose -f docker-compose.lite.yml down
        echo.
    ) else (
        echo.
        echo ERROR: Failed to start SmartRAG Lightweight
        echo Check the logs with: docker-compose -f docker-compose.lite.yml logs
    )
) else (
    echo Invalid choice. Please run the script again.
    pause
    exit /b 1
)

pause
