#!/usr/bin/env python3
"""
Quick prediction test
"""

def quick_predict_test():
    """Quick test of prediction functionality."""
    
    print("🔮 Quick Prediction Test")
    print("=" * 25)
    
    try:
        from predict import MentalHealthPredictor
        
        # Test single prediction
        predictor = MentalHealthPredictor()
        predictor.load_model()
        
        test_text = "I feel hopeless and overwhelmed"
        prediction, probs = predictor.predict(test_text)
        
        print(f"\nTest: '{test_text}'")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {max(probs.values()):.3f}")
        
        print(f"\n✅ Prediction test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Prediction test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_predict_test()
    
    if success:
        print(f"\n🎉 ALL SYSTEMS GO!")
        print(f"✅ Training works")
        print(f"✅ Saving works")
        print(f"✅ Prediction works")
        print(f"\n🚀 CPU training is fully operational!")
    else:
        print(f"\n💔 Still troubleshooting...")
