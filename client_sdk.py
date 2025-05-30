#!/usr/bin/env python3
"""
Mental Health Classifier - Python Client SDK
Easy-to-use client for the mental health classification API
"""

import requests
import json
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Result from a single prediction."""
    predicted_class: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    safety_flag: Optional[str] = None
    timestamp: Optional[datetime] = None

@dataclass
class BatchPredictionResult:
    """Result from batch prediction."""
    predictions: List[PredictionResult]
    total_processed: int
    high_risk_count: int
    timestamp: Optional[datetime] = None

class MentalHealthClassifierClient:
    """Client for Mental Health Classifier API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API server
            api_key: Optional API key for authentication (future use)
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'MentalHealthClassifier-Client/1.0.0'
        })
        
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}'
            })
    
    def health_check(self) -> Dict:
        """Check if the API is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get detailed model information."""
        try:
            response = self.session.get(f"{self.base_url}/model-info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get model info: {e}")
            raise
    
    def predict(
        self, 
        text: str, 
        include_probabilities: bool = True,
        confidence_threshold: float = 0.6
    ) -> PredictionResult:
        """
        Classify a single text.
        
        Args:
            text: Text to classify
            include_probabilities: Whether to include class probabilities
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            PredictionResult with classification results
        """
        try:
            payload = {
                "text": text,
                "include_probabilities": include_probabilities,
                "confidence_threshold": confidence_threshold
            }
            
            response = self.session.post(f"{self.base_url}/predict", json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            return PredictionResult(
                predicted_class=data['predicted_class'],
                confidence=data['confidence'],
                probabilities=data.get('probabilities'),
                safety_flag=data.get('safety_flag'),
                timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')) if data.get('timestamp') else None
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(
        self,
        texts: List[str],
        include_probabilities: bool = True,
        confidence_threshold: float = 0.6
    ) -> BatchPredictionResult:
        """
        Classify multiple texts in batch.
        
        Args:
            texts: List of texts to classify
            include_probabilities: Whether to include class probabilities
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            BatchPredictionResult with all classification results
        """
        try:
            payload = {
                "texts": texts,
                "include_probabilities": include_probabilities,
                "confidence_threshold": confidence_threshold
            }
            
            response = self.session.post(f"{self.base_url}/batch-predict", json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            predictions = []
            for pred_data in data['predictions']:
                predictions.append(PredictionResult(
                    predicted_class=pred_data['predicted_class'],
                    confidence=pred_data['confidence'],
                    probabilities=pred_data.get('probabilities'),
                    safety_flag=pred_data.get('safety_flag'),
                    timestamp=datetime.fromisoformat(pred_data['timestamp'].replace('Z', '+00:00')) if pred_data.get('timestamp') else None
                ))
            
            return BatchPredictionResult(
                predictions=predictions,
                total_processed=data['total_processed'],
                high_risk_count=data['high_risk_count'],
                timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')) if data.get('timestamp') else None
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def classify_high_risk(self, text: str) -> bool:
        """
        Quick check if text is classified as high risk (suicide).
        
        Args:
            text: Text to check
            
        Returns:
            True if classified as high risk, False otherwise
        """
        try:
            result = self.predict(text, include_probabilities=False)
            return result.safety_flag == 'HIGH_RISK' or result.predicted_class == 'Suicide'
        except Exception as e:
            logger.error(f"High risk check failed: {e}")
            return False  # Default to safe
    
    def get_confidence_distribution(self, texts: List[str]) -> Dict[str, int]:
        """
        Get distribution of confidence levels across multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with confidence level counts
        """
        try:
            results = self.predict_batch(texts, include_probabilities=False)
            
            distribution = {
                'high_confidence': 0,    # >= 0.8
                'medium_confidence': 0,  # 0.6 - 0.8
                'low_confidence': 0,     # < 0.6
                'high_risk': 0          # Suicide classifications
            }
            
            for pred in results.predictions:
                if pred.safety_flag == 'HIGH_RISK':
                    distribution['high_risk'] += 1
                elif pred.confidence >= 0.8:
                    distribution['high_confidence'] += 1
                elif pred.confidence >= 0.6:
                    distribution['medium_confidence'] += 1
                else:
                    distribution['low_confidence'] += 1
            
            return distribution
            
        except Exception as e:
            logger.error(f"Confidence distribution analysis failed: {e}")
            raise

# Convenience functions
def quick_classify(text: str, api_url: str = "http://localhost:8000") -> str:
    """
    Quick classification of a single text.
    
    Args:
        text: Text to classify
        api_url: API server URL
        
    Returns:
        Predicted class name
    """
    client = MentalHealthClassifierClient(api_url)
    result = client.predict(text, include_probabilities=False)
    return result.predicted_class

def is_high_risk(text: str, api_url: str = "http://localhost:8000") -> bool:
    """
    Quick check if text indicates high risk.
    
    Args:
        text: Text to check
        api_url: API server URL
        
    Returns:
        True if high risk, False otherwise
    """
    client = MentalHealthClassifierClient(api_url)
    return client.classify_high_risk(text)

# Example usage
if __name__ == "__main__":
    # Basic usage example
    client = MentalHealthClassifierClient()
    
    # Test connection
    try:
        health = client.health_check()
        print(f"API Status: {health['status']}")
        print(f"Model Accuracy: {health['model_accuracy']:.1%}")
    except Exception as e:
        print(f"API not available: {e}")
        exit(1)
    
    # Example predictions
    test_texts = [
        "I'm feeling a bit stressed about work",
        "I feel hopeless and want to end my life",
        "Having some anxiety about the presentation tomorrow",
        "This traffic is killing me slowly"
    ]
    
    print("\n=== Single Predictions ===")
    for text in test_texts:
        try:
            result = client.predict(text)
            print(f"Text: {text}")
            print(f"Prediction: {result.predicted_class} (confidence: {result.confidence:.3f})")
            if result.safety_flag:
                print(f"Safety Flag: {result.safety_flag}")
            print()
        except Exception as e:
            print(f"Prediction failed: {e}")
    
    print("\n=== Batch Prediction ===")
    try:
        batch_result = client.predict_batch(test_texts)
        print(f"Processed: {batch_result.total_processed} texts")
        print(f"High risk cases: {batch_result.high_risk_count}")
        
        for i, pred in enumerate(batch_result.predictions):
            print(f"{i+1}. {pred.predicted_class} ({pred.confidence:.3f})")
    except Exception as e:
        print(f"Batch prediction failed: {e}")
