# Deployment Guide

## Local Development Deployment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install package
pip install -e ".[dev]"

# Run tests
pytest

# Start development server
python -m llm_consortium.server
```

## Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY . .
RUN pip install ".[prod]"

CMD ["python", "-m", "llm_consortium.server"]
```

Build and run:
```bash
docker build -t llm-consortium .
docker run -p 8000:8000 llm-consortium
```

## Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-consortium
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-consortium
  template:
    metadata:
      labels:
        app: llm-consortium
    spec:
      containers:
      - name: llm-consortium
        image: llm-consortium:latest
        ports:
        - containerPort: 8000
        env:
        - name: LLM_USER_PATH
          value: "/data"
        volumeMounts:
        - name: data
          mountPath: "/data"
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: llm-consortium-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: llm-consortium
spec:
  selector:
    app: llm-consortium
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

Apply configuration:
```bash
kubectl apply -f kubernetes/deployment.yaml
```

## Cloud Deployment

### AWS Example

```yaml
# serverless.yml
service: llm-consortium

provider:
  name: aws
  runtime: python3.9
  region: us-east-1

functions:
  orchestrator:
    handler: llm_consortium.handler.lambda_handler
    events:
      - http:
          path: /orchestrate
          method: post
    environment:
      LLM_USER_PATH: /tmp
```

### Google Cloud Example

```yaml
# app.yaml
runtime: python39
entrypoint: gunicorn -b :$PORT llm_consortium.server:app

env_variables:
  LLM_USER_PATH: /tmp
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'llm-consortium'
    static_configs:
      - targets: ['localhost:8000']
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "LLM Consortium Metrics",
    "panels": [
      {
        "title": "Response Times",
        "type": "graph",
        "datasource": "Prometheus"
      }
    ]
  }
}
```

## Load Balancing

### NGINX Configuration

```nginx
upstream llm_consortium {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    location / {
        proxy_pass http://llm_consortium;
    }
}
```

## Backup and Recovery

```bash
# Backup script
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf backup_$DATE.tar.gz /data/llm_consortium
aws s3 cp backup_$DATE.tar.gz s3://backups/

# Recovery script
#!/bin/bash
aws s3 cp s3://backups/backup_$DATE.tar.gz .
tar -xzf backup_$DATE.tar.gz -C /data/
```

## Environment Configuration

```bash
# .env
LLM_USER_PATH=/data
LOG_LEVEL=INFO
MAX_WORKERS=4
CONFIDENCE_THRESHOLD=0.8
MAX_ITERATIONS=3
```

## Health Checks

```python
# health_check.py
from llm_consortium.health import check_health

def health_endpoint():
    status = check_health()
    return {"status": "healthy" if status else "unhealthy"}
```

## Security Configuration

```yaml
# security.yaml
rate_limiting:
  requests_per_minute: 60
  burst: 10

authentication:
  enabled: true
  jwt_secret: ${JWT_SECRET}

cors:
  allowed_origins:
    - https://example.com
  allowed_methods:
    - POST
    - GET
```
