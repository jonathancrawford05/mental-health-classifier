# 🚀 Mental Health Classifier - Docker Deployment Guide

## Quick Deploy (Docker)

### Prerequisites
- Docker Desktop installed
- 2GB+ available RAM
- Internet connection for downloading dependencies

### One-Command Deploy
```bash
# Clone and deploy
git clone <your-repo-url>
cd mental-health-classifier
docker build -t mental-health-classifier .
docker run -d --name mental-health-classifier -p 8000:8000 mental-health-classifier
```

### Verify Deployment
```bash
# Test the API
python test_api.py

# Or test manually
curl http://localhost:8000/health
```

## Expected Output
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_accuracy": 71.4,
  "version": "1.0.0"
}
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel anxious about everything"}'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I feel great", "I am worried", "I feel hopeless"]}'
```

## Model Classes
- **Anxiety**: Worry, panic, nervousness
- **Depression**: Sadness, hopelessness, low mood  
- **Suicide**: Self-harm ideation, suicide risk
- **Normal**: Regular expressions, non-clinical text

## Safety Features
- **HIGH_RISK**: Automatic flagging for suicide content
- **LOW_CONFIDENCE**: Flags uncertain predictions
- **REVIEW_RECOMMENDED**: Suggests manual review

## Performance Specs
- **Accuracy**: 71.4%
- **False Alarm Rate**: 4.8%
- **Vocabulary**: 365 tokens
- **Parameters**: 19.3M
- **Memory Usage**: ~500MB
- **Response Time**: <100ms per prediction

## Troubleshooting

### Common Issues

**1. NLTK Download Error**
```
Error: punkt_tab not found
```
**Solution**: This is fixed in the latest version. Ensure you're using the updated Dockerfile.

**2. Port Already in Use**
```
Error: Port 8000 already in use
```
**Solution**: 
```bash
# Use different port
docker run -d --name mental-health-classifier -p 8001:8000 mental-health-classifier

# Or stop existing container
docker stop mental-health-classifier
docker rm mental-health-classifier
```

**3. Model Loading Failed**
```
Error: Model not loaded
```
**Solution**: Check that `models/` directory contains required files:
- `best_model.pt`
- `vocab.pkl` 
- `model_info.json`

### Development Setup

For local development without Docker:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"

# Run server
python api_server.py
```

## Production Deployment

### Security Considerations
- API runs as non-root user in container
- No sensitive data exposed in logs
- Use reverse proxy (nginx) for HTTPS
- Implement rate limiting
- Monitor for abuse patterns

### Scaling
```bash
# Docker Compose for multiple instances
docker-compose up --scale app=3

# Or use Docker Swarm
docker service create --replicas 3 -p 8000:8000 mental-health-classifier
```

### Monitoring
- Health endpoint: `/health`
- Model info: `/model-info`
- API docs: `/docs`
- Logs: `docker logs mental-health-classifier`

## Support

- **Issues**: Create GitHub issue
- **Security**: Contact maintainers directly
- **Clinical Validation**: Consult healthcare professionals

⚠️ **Important**: This tool is for screening purposes only. Not a replacement for professional clinical assessment.
