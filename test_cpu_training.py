#!/usr/bin/env python3
"""
Test the fixed CPU training implementation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_cpu_training():
    """Test that the fixed CPU training works correctly."""
    
    print("🧪 TESTING FIXED CPU TRAINING")
    print("=" * 50)
    
    try:
        # Import the fixed training function
        from fixed_cpu_training import fixed_cpu_training
        
        # Run the training
        success, experiment_id, metrics = fixed_cpu_training()
        
        if success:
            print(f"\n✅ CPU training test PASSED!")
            print(f"   Experiment ID: {experiment_id}")
            print(f"   Best F1: {metrics.get('best_val_f1', 'N/A')}")
            
            # Test that model files exist
            from pathlib import Path
            models_dir = Path("models")
            
            required_files = [
                "best_model.pt",
                "vocab.pkl", 
                "model_info.json"
            ]
            
            print(f"\n📁 Checking model files:")
            all_files_exist = True
            for file in required_files:
                file_path = models_dir / file
                if file_path.exists():
                    print(f"   ✅ {file}")
                else:
                    print(f"   ❌ {file} - MISSING")
                    all_files_exist = False
            
            if all_files_exist:
                print(f"\n🎉 All model files saved correctly!")
                
                # Test inference
                print(f"\n🔮 Testing inference...")
                try:
                    from predict import MentalHealthPredictor
                    
                    predictor = MentalHealthPredictor()
                    predictor.load_model()
                    
                    test_text = "I feel hopeless and can't see any way forward"
                    prediction, probs = predictor.predict(test_text)
                    
                    print(f"   Test text: {test_text}")
                    print(f"   Prediction: {prediction}")
                    print(f"   Confidence: {max(probs.values()):.3f}")
                    print(f"   ✅ Inference test PASSED!")
                    
                    return True
                    
                except Exception as e:
                    print(f"   ❌ Inference test FAILED: {e}")
                    return False
            else:
                print(f"\n❌ Model files missing - CPU training incomplete")
                return False
                
        else:
            print(f"\n❌ CPU training test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_integration():
    """Test that main.py works with CPU mode."""
    
    print(f"\n🧪 TESTING MAIN.PY INTEGRATION")
    print("=" * 50)
    
    try:
        # Test main.py with CPU mode
        import subprocess
        import os
        
        # Change to project directory
        os.chdir(project_root)
        
        # Run main.py with CPU mode and sample data
        result = subprocess.run([
            sys.executable, "main.py", 
            "--mode", "simple-cpu",
            "--create-sample-data",
            "--sample-size", "500",
            "--seed", "42"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ Main.py CPU mode test PASSED!")
            print(f"   Output preview:")
            lines = result.stdout.split('\n')
            for line in lines[-10:]:
                if line.strip():
                    print(f"   {line}")
            return True
        else:
            print(f"❌ Main.py CPU mode test FAILED!")
            print(f"   Exit code: {result.returncode}")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Main.py test timed out")
        return False
    except Exception as e:
        print(f"❌ Main.py test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE CPU TRAINING TEST")
    print("=" * 60)
    
    # Test 1: Fixed CPU training function
    test1_passed = test_cpu_training()
    
    # Test 2: Main.py integration (only if test 1 passed)
    test2_passed = False
    if test1_passed:
        test2_passed = test_main_integration()
    else:
        print(f"\n⏭️ Skipping main.py test due to CPU training failure")
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"TEST SUMMARY")
    print(f"=" * 60)
    print(f"CPU Training Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Main.py Integration: {'✅ PASSED' if test2_passed else '❌ FAILED' if test1_passed else '⏭️ SKIPPED'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   CPU training is working correctly")
        print(f"   Model saving is fixed")
        print(f"   Inference pipeline works")
        print(f"   Main.py integration complete")
        print(f"\n✅ CPU training is ready as the default!")
    elif test1_passed:
        print(f"\n🟡 PARTIAL SUCCESS")
        print(f"   CPU training works")
        print(f"   Main.py integration needs attention")
    else:
        print(f"\n❌ CPU TRAINING STILL BROKEN")
        print(f"   Review error messages above")
