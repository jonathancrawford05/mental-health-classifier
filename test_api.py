#!/usr/bin/env python3
"""
Quick API test script for Mental Health Classifier
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_single_prediction():
    """Test single text classification."""
    print("=== Testing Single Prediction ===")
    
    test_texts = [
        "I feel anxious and worried about everything",
        "I have been feeling very depressed lately", 
        "I want to end my life, I can't take this anymore",
        "I am feeling great and optimistic today"
    ]
    
    for text in test_texts:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={
                "text": text,
                "include_probabilities": True,
                "confidence_threshold": 0.6
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nText: {text[:50]}...")
            print(f"Prediction: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.3f}")
            if result.get('safety_flag'):
                print(f"⚠️  Safety Flag: {result['safety_flag']}")
            print(f"Probabilities: {json.dumps(result['probabilities'], indent=2)}")
        else:
            print(f"Error: {response.status_code} - {response.text}")

def test_batch_prediction():
    """Test batch classification."""
    print("\n=== Testing Batch Prediction ===")
    
    batch_texts = [
        "I feel great today",
        "I am very anxious about my presentation",
        "I feel hopeless and depressed",
        "I want to hurt myself"
    ]
    
    response = requests.post(
        f"{BASE_URL}/batch-predict",
        json={
            "texts": batch_texts,
            "include_probabilities": True
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Processed {result['total_processed']} texts")
        print(f"High risk cases: {result['high_risk_count']}")
        
        for i, prediction in enumerate(result['predictions']):
            print(f"\nText {i+1}: {batch_texts[i][:30]}...")
            print(f"  → {prediction['predicted_class']} ({prediction['confidence']:.3f})")
            if prediction.get('safety_flag'):
                print(f"  ⚠️  {prediction['safety_flag']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_health_and_info():
    """Test health and model info endpoints."""
    print("\n=== Testing Health & Model Info ===")
    
    # Health check
    health_response = requests.get(f"{BASE_URL}/health")
    if health_response.status_code == 200:
        health = health_response.json()
        print(f"Status: {health['status']}")
        print(f"Model Loaded: {health['model_loaded']}")
        print(f"Model Accuracy: {health['model_accuracy']:.1%}")
    
    # Model info
    info_response = requests.get(f"{BASE_URL}/model-info")
    if info_response.status_code == 200:
        info = info_response.json()
        print(f"\nModel: {info['model_name']}")
        print(f"Classes: {info['classes']}")
        print(f"Vocabulary Size: {info['vocabulary_size']:,}")
        print(f"Parameters: {info['parameters']:,}")

if __name__ == "__main__":
    try:
        # Test health first
        test_health_and_info()
        
        # Test predictions
        test_single_prediction()
        test_batch_prediction()
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API at http://localhost:8000")
        print("   Make sure the Docker container is running!")
    except Exception as e:
        print(f"❌ Error: {e}")
