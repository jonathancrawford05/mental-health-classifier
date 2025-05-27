#!/usr/bin/env python3
"""
Simple test runner for minimal CPU training
"""

def run_test():
    """Run the minimal CPU training test directly."""
    
    print("🧪 Testing Minimal CPU Training...")
    print("=" * 40)
    
    try:
        # Import and run the minimal training
        from minimal_cpu_training import minimal_cpu_training, test_inference
        
        # Run training
        print("Step 1: Running CPU training...")
        training_success, metrics = minimal_cpu_training()
        
        if training_success:
            print(f"\n✅ Step 1 PASSED - Training successful!")
            print(f"   Best F1: {metrics['best_val_f1']:.3f}")
            
            # Check if files exist
            from pathlib import Path
            models_dir = Path("models")
            
            required_files = ["best_model.pt", "vocab.pkl", "model_info.json"]
            files_exist = all((models_dir / f).exists() for f in required_files)
            
            if files_exist:
                print(f"✅ Step 2 PASSED - All model files saved correctly")
                
                # Test inference
                print(f"\nStep 3: Testing inference...")
                inference_success = test_inference()
                
                if inference_success:
                    print(f"\n🎉 ALL TESTS PASSED!")
                    print(f"✅ CPU training works perfectly")
                    print(f"✅ Model saving is fixed")
                    print(f"✅ Inference pipeline works")
                    print(f"\n🎯 CPU training is ready to be the default!")
                    return True
                else:
                    print(f"\n🟡 Training works but inference has issues")
                    return False
            else:
                print(f"❌ Step 2 FAILED - Model files not saved properly")
                missing = [f for f in required_files if not (models_dir / f).exists()]
                print(f"   Missing files: {missing}")
                return False
        else:
            print(f"\n❌ Step 1 FAILED - Training unsuccessful")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_test()
    
    if success:
        print(f"\n🚀 Ready to integrate with main system!")
        print(f"   Next steps:")
        print(f"   1. python fixed_cpu_training.py --make-default")
        print(f"   2. python main.py --create-sample-data")
        print(f"   3. python predict.py --interactive")
    else:
        print(f"\n💔 Still needs work - check errors above")
