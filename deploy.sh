#!/bin/bash
# Quick Deployment Script for Mental Health Classifier API
# Builds and runs the API server with proper setup

echo "🚀 Mental Health Classifier - Quick Deployment"
echo "=============================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if model files exist
if [ ! -f "models/best_model.pt" ]; then
    print_error "Model file not found: models/best_model.pt"
    echo "Please ensure your trained model is in the models/ directory"
    exit 1
fi

if [ ! -f "models/vocab.pkl" ]; then
    print_error "Vocabulary file not found: models/vocab.pkl"
    echo "Please ensure your vocabulary file is in the models/ directory"
    exit 1
fi

print_status "Model files found"

# Check deployment option
if [ "$1" = "docker" ]; then
    echo ""
    echo "🐳 Docker Deployment"
    echo "===================="
    
    # Build Docker image
    print_status "Building Docker image..."
    docker build -t mental-health-classifier .
    
    if [ $? -ne 0 ]; then
        print_error "Docker build failed"
        exit 1
    fi
    
    # Run Docker container
    print_status "Starting Docker container..."
    docker run -d \
        --name mental-health-api \
        -p 8000:8000 \
        --restart unless-stopped \
        mental-health-classifier
    
    if [ $? -ne 0 ]; then
        print_error "Docker run failed"
        exit 1
    fi
    
    print_status "Docker container started successfully"
    
elif [ "$1" = "compose" ]; then
    echo ""
    echo "🐳 Docker Compose Deployment"
    echo "============================="
    
    # Check if docker-compose exists
    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose not found. Please install docker-compose"
        exit 1
    fi
    
    # Start with docker-compose
    print_status "Starting services with docker-compose..."
    docker-compose up -d
    
    if [ $? -ne 0 ]; then
        print_error "Docker compose failed"
        exit 1
    fi
    
    print_status "Services started successfully"
    
else
    echo ""
    echo "💻 Local Development Deployment"
    echo "==============================="
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
    if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 0 ]]; then
        print_warning "Python 3.8+ recommended, found Python $python_version"
    fi
    
    # Install dependencies
    print_status "Installing dependencies..."
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        print_error "Failed to install dependencies"
        exit 1
    fi
    
    # Start the API server
    print_status "Starting API server..."
    python api_server.py &
    
    # Get the PID
    API_PID=$!
    echo $API_PID > api_server.pid
    
    print_status "API server started with PID: $API_PID"
fi

# Wait for server to start
echo ""
echo "⏳ Waiting for server to start..."
sleep 5

# Test the API
echo ""
echo "🧪 Testing API..."

# Basic health check
if curl -s http://localhost:8000/health > /dev/null; then
    print_status "API is responding"
    
    # Get health info
    health_info=$(curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "Health Status:"
        echo "$health_info" | grep -E "(status|model_loaded|model_accuracy)" | sed 's/^/  /'
    fi
    
    # Quick prediction test
    echo ""
    echo "Testing prediction..."
    test_result=$(curl -s -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"text": "I am feeling anxious about work", "include_probabilities": false}')
    
    if [ $? -eq 0 ]; then
        predicted_class=$(echo "$test_result" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('predicted_class', 'Unknown'))" 2>/dev/null)
        confidence=$(echo "$test_result" | python3 -c "import sys, json; data = json.load(sys.stdin); print(f'{data.get(\"confidence\", 0):.3f}')" 2>/dev/null)
        
        if [ -n "$predicted_class" ] && [ "$predicted_class" != "Unknown" ]; then
            print_status "Prediction test successful: $predicted_class (confidence: $confidence)"
        else
            print_warning "Prediction test returned unexpected result"
        fi
    else
        print_warning "Prediction test failed"
    fi
    
else
    print_error "API is not responding on http://localhost:8000"
    echo "Check the logs for errors"
    exit 1
fi

# Display success information
echo ""
echo "🎉 DEPLOYMENT SUCCESSFUL!"
echo "========================="
echo ""
echo "🌐 API Endpoints:"
echo "   Health Check:    http://localhost:8000/health"
echo "   API Docs:        http://localhost:8000/docs"
echo "   ReDoc:           http://localhost:8000/redoc"
echo "   Single Predict:  POST http://localhost:8000/predict"
echo "   Batch Predict:   POST http://localhost:8000/batch-predict"
echo ""
echo "📊 Model Performance:"
echo "   Overall Accuracy: 71.4%"
echo "   False Alarm Rate: 4.8%"
echo "   Normal Precision: 91.7%"
echo "   Classes: Anxiety, Depression, Suicide, Normal"
echo ""
echo "🧪 Quick Tests:"
echo "   python test_api.py quick              # Quick functionality test"
echo "   python test_api.py                    # Full test suite"
echo "   python client_sdk.py                 # SDK example"
echo ""
echo "🔧 Management:"
if [ "$1" = "docker" ]; then
    echo "   docker logs mental-health-api         # View logs"
    echo "   docker stop mental-health-api         # Stop container"
    echo "   docker start mental-health-api        # Start container"
elif [ "$1" = "compose" ]; then
    echo "   docker-compose logs                   # View logs"
    echo "   docker-compose stop                   # Stop services"
    echo "   docker-compose start                  # Start services"
else
    echo "   kill $(cat api_server.pid)            # Stop server"
    echo "   tail -f api_server.log                # View logs (if logging enabled)"
fi

echo ""
echo "🚀 Your 71.4% accuracy mental health classifier is now live!"
echo "Ready for production integration and testing."
