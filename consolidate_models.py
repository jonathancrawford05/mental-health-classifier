#!/usr/bin/env python3
"""
Final cleanup script for mental health classifier experiments
Run this to complete the consolidation process
"""

import shutil
import os
from pathlib import Path

def cleanup_experiments():
    """Remove debug and test experiment directories"""
    
    base_path = Path("/Users/family_crawfords/projects/claude-mcp/mental-health-classifier/experiments")
    
    # Directories to remove
    dirs_to_remove = [
        # Ultra micro debug runs
        "baseline_ultra_micro_20250525_111914",
        "baseline_ultra_micro_20250525_112542", 
        "baseline_ultra_micro_20250525_112939",
        "baseline_ultra_micro_20250525_174721",
        "baseline_ultra_micro_20250525_175441",
        
        # Excess micro debug runs (keeping one for reference)
        "baseline_micro_20250525_110829",
        "baseline_micro_20250525_111655",
        
        # Duplicate baseline small runs  
        "baseline_small_20250525_072841",
        "baseline_small_20250525_110000",
        
        # Test and mock runs
        "mock_test_20250525_112344",
        "safe_test_20250525_112320", 
        "minimal_test_20250525_115149",
        
        # Additional baseline runs
        "medium_baseline_20250525_115743",
        "micro_baseline_20250525_115448",
        "small_baseline_20250525_115549", 
        "ultra_micro_production_20250525_115514",
        "ultra_micro_test_20250525_114923",
        
        # Absolute minimal (if not needed)
        "absolute_minimal",
    ]
    
    print("🧹 Starting experiment cleanup...")
    print(f"Target directory: {base_path}")
    
    removed = []
    not_found = []
    errors = []
    
    for dirname in dirs_to_remove:
        dir_path = base_path / dirname
        
        if not dir_path.exists():
            not_found.append(dirname)
            continue
            
        try:
            # Get size before removal (rough estimate)
            total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            
            shutil.rmtree(dir_path)
            removed.append((dirname, size_mb))
            print(f"✅ Removed: {dirname} ({size_mb:.1f} MB)")
            
        except Exception as e:
            errors.append((dirname, str(e)))
            print(f"❌ Error removing {dirname}: {e}")
    
    # Summary
    print(f"\n📊 Cleanup Summary:")
    print(f"✅ Successfully removed: {len(removed)} directories")
    print(f"⚠️  Not found: {len(not_found)} directories") 
    print(f"❌ Errors: {len(errors)} directories")
    
    total_mb = sum(size for _, size in removed)
    print(f"💾 Space freed: {total_mb:.1f} MB ({total_mb/1024:.1f} GB)")
    
    if errors:
        print(f"\n❌ Errors encountered:")
        for dirname, error in errors:
            print(f"   {dirname}: {error}")
    
    # Show remaining experiments
    remaining = [d for d in os.listdir(base_path) 
                if (base_path / d).is_dir() and not d.startswith('.')]
    
    print(f"\n📁 Remaining experiments ({len(remaining)}):")
    for exp in sorted(remaining):
        print(f"   - {exp}")
    
    return len(removed), len(errors)

def main():
    print("🏥 Mental Health Classifier - Model Consolidation")
    print("=" * 50)
    
    response = input("This will permanently delete debug experiment directories. Continue? (y/N): ")
    
    if response.lower() != 'y':
        print("Cleanup cancelled.")
        return
    
    removed_count, error_count = cleanup_experiments()
    
    if error_count == 0:
        print(f"\n🎉 Consolidation complete! Removed {removed_count} debug experiments.")
        print("\n📋 Next steps:")
        print("   1. Review models/production/ for your production-ready models")
        print("   2. Check experiments/archived/ for reference implementations") 
        print("   3. Use models_registry_consolidated.json for model metadata")
        print(f"   4. Consider retraining medium_v1 model for better depression class performance")
    else:
        print(f"\n⚠️  Consolidation completed with {error_count} errors.")
        print("   Please review the errors above and resolve manually if needed.")

if __name__ == "__main__":
    main()
