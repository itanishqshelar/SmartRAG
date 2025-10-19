# Docker Deployment Guide for SmartRAG

This guide provides instructions for running SmartRAG using Docker.

## 📦 Deployment Options

### Option 1: Full Stack (Recommended for Production)

Includes both Streamlit app and Ollama in one container.

### Option 2: Lightweight (Recommended for Development)

Streamlit in Docker, Ollama runs on host machine.

---

## 🚀 Quick Start

### Prerequisites

- Docker Desktop installed
- Docker Compose installed
- At least 8GB RAM available
- 20GB free disk space (for models)

### Option 1: Full Stack Deployment

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Access the application
# Open http://localhost:8501 in your browser
```

**First Run Notes:**

- Initial startup takes 10-15 minutes (downloading models)
- Llama 3.1 8B model: ~4.7GB
- Nomic Embed Text model: ~274MB
- Models are cached in Docker volume for subsequent runs

### Option 2: Lightweight Deployment

```bash
# First, ensure Ollama is running on your host
ollama serve

# In another terminal, pull required models
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Build and start the lightweight container
docker-compose -f docker-compose.lite.yml up -d

# Access the application
# Open http://localhost:8501 in your browser
```

---

## 🔧 Configuration

### Environment Variables

You can customize the deployment by setting environment variables in `docker-compose.yml`:

```yaml
environment:
  - STREAMLIT_SERVER_PORT=8501
  - STREAMLIT_SERVER_ADDRESS=0.0.0.0
  - OLLAMA_HOST=http://localhost:11434
```

### Volume Mounts

Data persistence is handled through Docker volumes:

- `./vector_db` - ChromaDB vector database
- `./user_data` - User session data
- `./temp_uploads` - Temporary file uploads
- `./logs` - Application logs
- `./file_storage.db` - SQLite database
- `ollama_models` - Ollama model cache (Docker volume)

---

## 🛠️ Build from Source

### Build Full Stack Image

```bash
docker build -t smartrag:latest -f Dockerfile .
```

### Build Lightweight Image

```bash
docker build -t smartrag:lite -f Dockerfile.lite .
```

### Run Manually

```bash
# Full stack
docker run -d \
  -p 8501:8501 \
  -p 11434:11434 \
  -v $(pwd)/vector_db:/app/vector_db \
  -v $(pwd)/user_data:/app/user_data \
  --name smartrag \
  smartrag:latest

# Lightweight (with host Ollama)
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/vector_db:/app/vector_db \
  -v $(pwd)/user_data:/app/user_data \
  --add-host host.docker.internal:host-gateway \
  --name smartrag-lite \
  smartrag:lite
```

---

## 📊 Resource Requirements

### Minimum Requirements

- **CPU**: 4 cores
- **RAM**: 8GB
- **Disk**: 20GB free space

### Recommended Requirements

- **CPU**: 8+ cores
- **RAM**: 16GB
- **Disk**: 50GB free space (for multiple models and data)
- **GPU**: Optional (for faster inference)

### Memory Allocation

Adjust memory limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 8G # Maximum memory
    reservations:
      memory: 4G # Guaranteed memory
```

---

## 🔍 Monitoring & Debugging

### View Logs

```bash
# Follow logs in real-time
docker-compose logs -f

# View specific service logs
docker-compose logs -f smartrag

# View last 100 lines
docker-compose logs --tail=100
```

### Check Container Status

```bash
# List running containers
docker-compose ps

# Check health status
docker inspect --format='{{json .State.Health}}' smartrag-app
```

### Access Container Shell

```bash
# Execute bash inside container
docker-compose exec smartrag bash

# Check if Ollama is running
docker-compose exec smartrag curl http://localhost:11434/api/tags
```

### Common Issues

**Issue: Ollama models not downloading**

```bash
# Enter container and manually pull models
docker-compose exec smartrag bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

**Issue: Out of memory**

```bash
# Increase memory limits in docker-compose.yml
# Or reduce model size to llama3.1:7b
```

**Issue: Connection refused to Ollama**

```bash
# Check if Ollama service is running
docker-compose exec smartrag ps aux | grep ollama

# Restart the container
docker-compose restart
```

---

## 🔄 Updates & Maintenance

### Update Application

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Backup Data

```bash
# Backup vector database and user data
tar -czf smartrag-backup-$(date +%Y%m%d).tar.gz \
  vector_db/ user_data/ file_storage.db

# Backup Ollama models (optional)
docker run --rm \
  -v smartrag_ollama_models:/models \
  -v $(pwd):/backup \
  alpine tar -czf /backup/ollama-models.tar.gz -C /models .
```

### Restore Data

```bash
# Restore from backup
tar -xzf smartrag-backup-YYYYMMDD.tar.gz
docker-compose up -d
```

---

## 🌐 Production Deployment

### Using Nginx as Reverse Proxy

```nginx
server {
    listen 80;
    server_name smartrag.yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Enable HTTPS with Let's Encrypt

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d smartrag.yourdomain.com
```

### Docker Compose for Production

```yaml
version: "3.8"

services:
  smartrag:
    build: .
    restart: always
    environment:
      - STREAMLIT_SERVER_PORT=8501
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 8G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

## 🧪 Testing Deployment

```bash
# Health check
curl http://localhost:8501/_stcore/health

# Test Ollama
curl http://localhost:11434/api/tags

# Test application
# Upload a test file through the web interface at http://localhost:8501
```

---

## 🛑 Stopping & Cleanup

### Stop Containers

```bash
# Stop services
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v
```

### Remove Images

```bash
# Remove SmartRAG images
docker rmi smartrag:latest smartrag:lite

# Clean up unused images
docker image prune -a
```

### Complete Cleanup

```bash
# Remove everything (containers, images, volumes)
docker-compose down -v --rmi all
docker system prune -a --volumes
```

---

## 📝 Tips & Best Practices

1. **First Run**: Allow 15-20 minutes for initial model downloads
2. **Persistence**: Always use volume mounts for data persistence
3. **Resources**: Monitor memory usage with `docker stats`
4. **Security**: Run in production behind a reverse proxy with SSL
5. **Backups**: Regularly backup vector_db and user_data directories
6. **Updates**: Pull new models periodically for improved performance
7. **Logs**: Set up log rotation to prevent disk space issues

---

## 🔗 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [SmartRAG GitHub](https://github.com/itanishqshelar/SmartRAG)
