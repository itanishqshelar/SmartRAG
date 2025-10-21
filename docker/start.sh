#!/bin/bash
# SmartRAG Docker Quick Start Script for Linux/Mac

set -e

echo "========================================"
echo "SmartRAG Docker Deployment"
echo "========================================"
echo

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

echo "Docker is installed. Checking Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: Docker Compose is not installed"
    echo "Please install Docker Compose from https://docs.docker.com/compose/install/"
    exit 1
fi

echo
echo "Select deployment option:"
echo "1. Full Stack (App + Ollama in Docker) - Recommended"
echo "2. Lightweight (App in Docker, Ollama on host)"
echo
read -p "Enter your choice (1 or 2): " choice

if [ "$choice" == "1" ]; then
    echo
    echo "Starting Full Stack deployment..."
    echo "This will take 10-15 minutes on first run (downloading models)"
    echo
    docker-compose up -d
    
    echo
    echo "========================================"
    echo "SmartRAG is starting!"
    echo "========================================"
    echo
    echo "Access the application at: http://localhost:8501"
    echo
    echo "To view logs: docker-compose logs -f"
    echo "To stop: docker-compose down"
    echo
    
elif [ "$choice" == "2" ]; then
    echo
    echo "Starting Lightweight deployment..."
    echo
    echo "IMPORTANT: Make sure Ollama is running on your host machine!"
    echo "If not running, open another terminal and run: ollama serve"
    echo
    read -p "Press Enter to continue..."
    
    docker-compose -f docker-compose.lite.yml up -d
    
    echo
    echo "========================================"
    echo "SmartRAG Lightweight is running!"
    echo "========================================"
    echo
    echo "Access the application at: http://localhost:8501"
    echo
    echo "To view logs: docker-compose -f docker-compose.lite.yml logs -f"
    echo "To stop: docker-compose -f docker-compose.lite.yml down"
    echo
    
else
    echo "Invalid choice. Please run the script again."
    exit 1
fi
