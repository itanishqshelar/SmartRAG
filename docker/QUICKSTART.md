# 🐳 SmartRAG Docker Quick Start

This folder contains all Docker-related files for SmartRAG deployment.

## 📁 Contents

```
docker/
├── README.md                   # Comprehensive Docker deployment guide
├── Dockerfile                  # Full stack image (Streamlit + Ollama)
├── Dockerfile.lite             # Lightweight image (Streamlit only)
├── docker-compose.yml          # Full stack orchestration
├── docker-compose.lite.yml     # Lightweight orchestration
├── .dockerignore              # Docker build ignore rules
├── start.bat                   # Windows quick start script
└── start.sh                    # Linux/Mac quick start script
```

## 🚀 Quick Start

### Windows

```powershell
cd docker
.\start.bat
```

### Linux/Mac

```bash
cd docker
chmod +x start.sh
./start.sh
```

### Manual Start

```bash
cd docker
docker-compose up -d
```

Access the application at: **http://localhost:8501**

## 📚 Full Documentation

See [README.md](README.md) in this folder for:

- Detailed deployment options
- Configuration guide
- Troubleshooting
- Production setup
- Backup and restore

## ⚡ Quick Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild
docker-compose up -d --build
```

## 🔧 Requirements

- Docker Desktop installed
- 8GB+ RAM available
- 20GB free disk space
- Ports 8501 and 11434 available

---

**For complete instructions, see [README.md](README.md)**
