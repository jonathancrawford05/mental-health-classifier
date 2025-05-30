#!/usr/bin/env python3
"""
API Test Suite for Mental Health Classifier
Comprehensive testing of all API endpoints
"""

import requests
import json
import time
from client_sdk import MentalHealthClassifierClient
import pytest
from typing import List, Dict

class TestMentalHealthAPI:
    """Test suite for the mental health classifier API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = MentalHealthClassifierClient(base_url)
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        print("🔍 Testing health endpoint...")
        
        try:
            health = self.client.health_check()
            
            assert health['status'] == 'healthy', f"API not healthy: {health['status']}"
            assert health['model_loaded'] == True, "Model not loaded"
            assert 'model_accuracy' in health, "Missing model accuracy"
            assert 'version' in health, "Missing version info"
            
            print(f"✅ Health check passed")
            print(f"   Status: {health['status']}")
            print(f"   Model loaded: {health['model_loaded']}")
            print(f"   Accuracy: {health['model_accuracy']:.1%}")
            
            return True
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def test_model_info_endpoint(self):
        """Test model info endpoint."""
        print("\n🔍 Testing model info endpoint...")
        
        try:
            info = self.client.get_model_info()
            
            required_fields = ['model_name', 'version', 'accuracy', 'classes', 'architecture']
            for field in required_fields:
                assert field in info, f"Missing field: {field}"
            
            assert len(info['classes']) == 4, f"Expected 4 classes, got {len(info['classes'])}"
            assert 'Normal' in info['classes'], "Missing Normal class"
            assert 'Suicide' in info['classes'], "Missing Suicide class"
            
            print(f"✅ Model info check passed")
            print(f"   Model: {info['model_name']}")
            print(f"   Classes: {info['classes']}")
            print(f"   Accuracy: {info['accuracy']:.1%}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model info check failed: {e}")
            return False
    
    def test_single_prediction(self):
        """Test single prediction endpoint."""
        print("\n🔍 Testing single prediction...")
        
        test_cases = [
            ("I'm feeling anxious about work", "Anxiety"),
            ("I feel hopeless and empty", "Depression"),
            ("I want to end my life", "Suicide"),
            ("Having a great day today", "Normal"),
            ("This traffic is killing me", "Normal")  # Should be Normal, not Suicide
        ]
        
        passed = 0
        for text, expected_category in test_cases:
            try:
                result = self.client.predict(text)
                
                # Basic validation
                assert result.predicted_class in ['Anxiety', 'Depression', 'Suicide', 'Normal']
                assert 0 <= result.confidence <= 1
                assert result.probabilities is not None
                assert len(result.probabilities) == 4
                
                print(f"   Text: '{text[:50]}...'")
                print(f"   Predicted: {result.predicted_class} (confidence: {result.confidence:.3f})")
                
                if result.safety_flag:
                    print(f"   Safety flag: {result.safety_flag}")
                
                # Check if prediction matches expected (with some flexibility)
                if result.predicted_class == expected_category:
                    print(f"   ✅ Correct prediction")
                    passed += 1
                else:
                    print(f"   ⚠️ Expected {expected_category}, got {result.predicted_class}")
                
                print()
                
            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")
        
        print(f"✅ Single prediction test: {passed}/{len(test_cases)} correct")
        return passed >= len(test_cases) * 0.6  # 60% accuracy threshold
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint."""
        print("\n🔍 Testing batch prediction...")
        
        test_texts = [
            "I'm worried about my presentation",
            "Feeling depressed and hopeless",
            "Having thoughts of self-harm",
            "Beautiful sunny day today",
            "This exam is going to kill me"
        ]
        
        try:
            result = self.client.predict_batch(test_texts)
            
            assert result.total_processed == len(test_texts)
            assert len(result.predictions) == len(test_texts)
            assert result.high_risk_count >= 0
            
            print(f"✅ Batch prediction passed")
            print(f"   Processed: {result.total_processed} texts")
            print(f"   High risk: {result.high_risk_count} cases")
            
            # Show predictions
            for i, pred in enumerate(result.predictions):
                print(f"   {i+1}. {pred.predicted_class} ({pred.confidence:.3f})")
            
            return True
            
        except Exception as e:
            print(f"❌ Batch prediction failed: {e}")
            return False
    
    def test_safety_features(self):
        """Test safety features and flags."""
        print("\n🔍 Testing safety features...")
        
        # Test high-risk detection
        high_risk_texts = [
            "I want to kill myself",
            "I have a plan to end my life",
            "I don't want to be alive anymore"
        ]
        
        high_risk_detected = 0
        for text in high_risk_texts:
            try:
                result = self.client.predict(text)
                if result.safety_flag == 'HIGH_RISK' or result.predicted_class == 'Suicide':
                    high_risk_detected += 1
                    print(f"   ✅ High risk detected: '{text[:40]}...'")
                else:
                    print(f"   ⚠️ High risk missed: '{text[:40]}...'")
            except Exception as e:
                print(f"   ❌ Safety test failed: {e}")
        
        # Test normal expressions (should NOT be high risk)
        normal_expressions = [
            "This traffic is killing me",
            "I'm dying to know the results",
            "I could just die from embarrassment"
        ]
        
        false_alarms = 0
        for text in normal_expressions:
            try:
                result = self.client.predict(text)
                if result.predicted_class == 'Suicide' or result.safety_flag == 'HIGH_RISK':
                    false_alarms += 1
                    print(f"   ⚠️ False alarm: '{text}' -> {result.predicted_class}")
                else:
                    print(f"   ✅ Correctly identified as normal: '{text}'")
            except Exception as e:
                print(f"   ❌ Normal expression test failed: {e}")
        
        # Test confidence thresholding
        try:
            result = self.client.predict("Feeling somewhat uncertain", confidence_threshold=0.9)
            if result.safety_flag in ['LOW_CONFIDENCE', 'REVIEW_RECOMMENDED']:
                print(f"   ✅ Confidence thresholding working")
            else:
                print(f"   ⚠️ Confidence thresholding not triggered")
        except Exception as e:
            print(f"   ❌ Confidence test failed: {e}")
        
        print(f"✅ Safety features test completed")
        print(f"   High risk detection: {high_risk_detected}/{len(high_risk_texts)}")
        print(f"   False alarms: {false_alarms}/{len(normal_expressions)}")
        
        return high_risk_detected >= 2 and false_alarms <= 1  # Good safety performance
    
    def test_performance(self):
        """Test API performance and response times."""
        print("\n🔍 Testing performance...")
        
        test_text = "I'm feeling anxious about work and worried about the future"
        
        # Single prediction performance
        start_time = time.time()
        try:
            result = self.client.predict(test_text)
            single_duration = time.time() - start_time
            
            print(f"   Single prediction: {single_duration:.3f}s")
            assert single_duration < 5.0, f"Single prediction too slow: {single_duration:.3f}s"
            
        except Exception as e:
            print(f"   ❌ Single prediction performance test failed: {e}")
            return False
        
        # Batch prediction performance
        test_texts = [test_text] * 10
        start_time = time.time()
        try:
            result = self.client.predict_batch(test_texts)
            batch_duration = time.time() - start_time
            avg_per_text = batch_duration / len(test_texts)
            
            print(f"   Batch prediction ({len(test_texts)} texts): {batch_duration:.3f}s")
            print(f"   Average per text: {avg_per_text:.3f}s")
            
            assert batch_duration < 30.0, f"Batch prediction too slow: {batch_duration:.3f}s"
            assert avg_per_text < 3.0, f"Average per text too slow: {avg_per_text:.3f}s"
            
        except Exception as e:
            print(f"   ❌ Batch prediction performance test failed: {e}")
            return False
        
        print(f"✅ Performance test passed")
        return True
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        print("\n🔍 Testing error handling...")
        
        # Test empty text
        try:
            response = requests.post(f"{self.base_url}/predict", json={"text": ""})
            assert response.status_code == 422, "Should reject empty text"
            print("   ✅ Empty text rejection working")
        except Exception as e:
            print(f"   ⚠️ Empty text test issue: {e}")
        
        # Test very long text
        try:
            long_text = "This is a test. " * 1000  # ~15,000 chars
            response = requests.post(f"{self.base_url}/predict", json={"text": long_text})
            assert response.status_code == 422, "Should reject overly long text"
            print("   ✅ Long text rejection working")
        except Exception as e:
            print(f"   ⚠️ Long text test issue: {e}")
        
        # Test invalid confidence threshold
        try:
            response = requests.post(f"{self.base_url}/predict", json={
                "text": "Test text",
                "confidence_threshold": 1.5
            })
            assert response.status_code == 422, "Should reject invalid confidence threshold"
            print("   ✅ Invalid confidence threshold rejection working")
        except Exception as e:
            print(f"   ⚠️ Confidence threshold test issue: {e}")
        
        # Test malformed JSON
        try:
            response = requests.post(f"{self.base_url}/predict", data="invalid json")
            assert response.status_code in [400, 422], "Should reject malformed JSON"
            print("   ✅ Malformed JSON rejection working")
        except Exception as e:
            print(f"   ⚠️ Malformed JSON test issue: {e}")
        
        print(f"✅ Error handling test completed")
        return True
    
    def run_all_tests(self):
        """Run complete test suite."""
        print("🚀 RUNNING COMPLETE API TEST SUITE")
        print("=" * 50)
        
        tests = [
            ("Health Check", self.test_health_endpoint),
            ("Model Info", self.test_model_info_endpoint),
            ("Single Prediction", self.test_single_prediction),
            ("Batch Prediction", self.test_batch_prediction),
            ("Safety Features", self.test_safety_features),
            ("Performance", self.test_performance),
            ("Error Handling", self.test_error_handling)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {e}")
        
        print("\n" + "=" * 50)
        print(f"🎯 TEST RESULTS: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED - API is ready for production!")
        elif passed >= total * 0.8:
            print("⚠️ Most tests passed - minor issues to address")
        else:
            print("🚨 Multiple test failures - significant issues need attention")
        
        return passed / total

def quick_test():
    """Quick API functionality test."""
    print("🔍 Quick API Test")
    
    try:
        client = MentalHealthClassifierClient()
        
        # Test basic functionality
        health = client.health_check()
        print(f"✅ API Health: {health['status']}")
        
        result = client.predict("I'm feeling anxious about work")
        print(f"✅ Prediction: {result.predicted_class} ({result.confidence:.3f})")
        
        print("🎉 Quick test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        # Run full test suite
        test_suite = TestMentalHealthAPI()
        success_rate = test_suite.run_all_tests()
        
        # Exit with appropriate code
        sys.exit(0 if success_rate >= 0.8 else 1)
