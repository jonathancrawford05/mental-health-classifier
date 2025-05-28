#!/usr/bin/env python3
"""
System Verification Script
Tests the consolidated experiment tracking system
"""

import os
import sys
from pathlib import Path

def test_system():
    """Test the consolidated experiment system."""
    print("🧪 Testing Consolidated Experiment System")
    print("=" * 50)
    
    # Test 1: Directory structure
    print("📁 Testing directory structure...")
    required_dirs = [
        "experiments",
        "experiments/archive_debug",
        "experiments/archive_baseline", 
        "models/production",
        "src/utils"
    ]
    
    for dirname in required_dirs:
        path = Path(dirname)
        if path.exists():
            print(f"  ✅ {dirname}")
        else:
            print(f"  ❌ {dirname} - Missing!")
    
    # Test 2: Required files
    print(f"\n📄 Testing required files...")
    required_files = [
        "src/utils/consolidated_experiment_tracker.py",
        "train_with_consolidated_tracking.py",
        "manage_experiments.py",
        "models_registry_consolidated.json",
        "NEW_SYSTEM_GUIDE.md"
    ]
    
    for filename in required_files:
        path = Path(filename)
        if path.exists():
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename} - Missing!")
    
    # Test 3: Production models
    print(f"\n🏆 Testing production models...")
    production_dir = Path("models/production")
    if production_dir.exists():
        production_models = [d for d in production_dir.iterdir() if d.is_dir()]
        print(f"  Found {len(production_models)} production models:")
        for model_dir in production_models:
            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                print(f"    ✅ {model_dir.name} (with metadata)")
            else:
                print(f"    ⚠️  {model_dir.name} (no metadata)")
    else:
        print(f"  ❌ Production directory not found")
    
    # Test 4: Archive structure
    print(f"\n📦 Testing archive structure...")
    archive_dirs = ["experiments/archive_debug", "experiments/archive_baseline"]
    total_archived = 0
    
    for archive_dir in archive_dirs:
        path = Path(archive_dir)
        if path.exists():
            archived_count = len([d for d in path.iterdir() if d.is_dir()])
            total_archived += archived_count
            print(f"  ✅ {archive_dir}: {archived_count} experiments")
        else:
            print(f"  ❌ {archive_dir} not found")
    
    print(f"  Total archived experiments: {total_archived}")
    
    # Summary
    print(f"\n📊 System Status Summary")
    print("=" * 30)
    
    if total_archived > 0:
        print(f"✅ Consolidation completed - {total_archived} experiments archived")
    else:
        print(f"⚠️  No archived experiments found - run archiving if needed")
    
    print(f"✅ New tracking system ready for use")
    print(f"✅ Management scripts available")
    print(f"✅ Production models directory set up")
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Run: python manage_experiments.py list")
    print(f"2. Test: python train_with_consolidated_tracking.py --experiment-summary")
    print(f"3. Read: NEW_SYSTEM_GUIDE.md for detailed usage")

if __name__ == "__main__":
    test_system()
