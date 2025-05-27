#!/usr/bin/env python3
"""
Quick fix test - just run the fixed CPU training
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def quick_test():
    """Quick test of just the fixed CPU training."""
    
    print("🧪 QUICK CPU TRAINING TEST")
    print("=" * 40)
    
    try:
        from fixed_cpu_training import fixed_cpu_training
        
        print("Running fixed CPU training...")
        success, experiment_id, metrics = fixed_cpu_training()
        
        if success:
            print(f"\n✅ SUCCESS!")
            print(f"   Experiment: {experiment_id}")
            print(f"   F1 Score: {metrics.get('best_val_f1', 'N/A')}")
            
            # Check if model files exist
            from pathlib import Path
            models_dir = Path("models")
            
            files_to_check = ["best_model.pt", "vocab.pkl", "model_info.json"]
            print(f"\n📁 Model files:")
            for file in files_to_check:
                path = models_dir / file
                status = "✅" if path.exists() else "❌"
                print(f"   {status} {file}")
            
            return True
        else:
            print(f"\n❌ Training failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print(f"\n🎉 CPU training is working!")
        print(f"Next: python predict.py --text 'I feel sad'")
    else:
        print(f"\n💔 Still needs fixing")
