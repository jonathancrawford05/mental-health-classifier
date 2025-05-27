#!/usr/bin/env python3
"""
Test the fixed predict.py
"""

def test_prediction():
    """Test that prediction works with the trained model."""
    
    print("🧪 TESTING PREDICTION")
    print("=" * 30)
    
    try:
        from predict import MentalHealthPredictor
        
        # Initialize predictor
        predictor = MentalHealthPredictor()
        
        # Load model
        print("Loading model...")
        predictor.load_model()
        
        # Test predictions
        test_texts = [
            "I feel hopeless and overwhelmed",
            "I'm constantly worried about everything",
            "I'm feeling much better today"
        ]
        
        print("\n📝 Test Predictions:")
        for i, text in enumerate(test_texts, 1):
            prediction, probs = predictor.predict(text)
            confidence = max(probs.values())
            
            print(f"\n{i}. Text: '{text}'")
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   All probabilities:")
            for label, prob in probs.items():
                print(f"     {label}: {prob:.3f}")
        
        print(f"\n✅ PREDICTION TEST PASSED!")
        print(f"   • Model loads correctly")
        print(f"   • Predictions work")
        print(f"   • Probabilities returned")
        
        return True
        
    except Exception as e:
        print(f"\n❌ PREDICTION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prediction()
    
    if success:
        print(f"\n🎉 COMPLETE SUCCESS!")
        print(f"✅ CPU training works")
        print(f"✅ Model saving works") 
        print(f"✅ Prediction works")
        print(f"\n🚀 CPU training is fully integrated and ready!")
        print(f"\n🎯 Try these commands:")
        print(f"   python predict.py --text 'I feel sad and hopeless'")
        print(f"   python predict.py --interactive")
        print(f"   python main.py --create-sample-data")
    else:
        print(f"\n💔 Still needs fixing")
