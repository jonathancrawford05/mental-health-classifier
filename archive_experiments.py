#!/usr/bin/env python3
"""
Git-Safe Model Consolidation Script
Archives experiments instead of deleting them, with full documentation
"""

import json
import shutil
import os
from pathlib import Path
from datetime import datetime

def load_consolidation_log():
    """Load the consolidation log to get move instructions"""
    log_path = Path("experiment_consolidation_log.json")
    with open(log_path, 'r') as f:
        return json.load(f)

def archive_experiments_safely():
    """Archive experiments with full documentation"""
    
    print("🏥 Mental Health Classifier - Git-Safe Consolidation")
    print("=" * 60)
    
    base_path = Path("/Users/family_crawfords/projects/claude-mcp/mental-health-classifier")
    exp_path = base_path / "experiments"
    
    # Load consolidation plan
    try:
        log_data = load_consolidation_log()
    except Exception as e:
        print(f"❌ Error loading consolidation log: {e}")
        return
    
    # Create archive directories
    archive_debug = exp_path / "archive_debug"
    archive_baseline = exp_path / "archive_baseline" 
    
    archive_debug.mkdir(exist_ok=True)
    archive_baseline.mkdir(exist_ok=True)
    
    # Track moves for documentation
    moves_log = []
    errors_log = []
    
    print("📁 Creating archive structure...")
    
    # Archive debug experiments
    debug_experiments = log_data["consolidation_actions"]["pending_archive_moves"]["to_archive_debug"]
    print(f"\n🔧 Archiving {len(debug_experiments)} debug experiments...")
    
    for exp_name in debug_experiments:
        source = exp_path / exp_name
        dest = archive_debug / exp_name
        
        if source.exists():
            try:
                shutil.move(str(source), str(dest))
                moves_log.append({
                    "from": f"experiments/{exp_name}",
                    "to": f"experiments/archive_debug/{exp_name}",
                    "category": "debug",
                    "timestamp": datetime.now().isoformat()
                })
                print(f"  ✅ {exp_name}")
            except Exception as e:
                errors_log.append(f"Failed to move {exp_name}: {e}")
                print(f"  ❌ {exp_name}: {e}")
        else:
            print(f"  ⚠️  Not found: {exp_name}")
    
    # Archive baseline experiments  
    baseline_experiments = log_data["consolidation_actions"]["pending_archive_moves"]["to_archive_baseline"]
    print(f"\n📊 Archiving {len(baseline_experiments)} baseline experiments...")
    
    for exp_name in baseline_experiments:
        source = exp_path / exp_name
        dest = archive_baseline / exp_name
        
        if source.exists():
            try:
                shutil.move(str(source), str(dest))
                moves_log.append({
                    "from": f"experiments/{exp_name}",
                    "to": f"experiments/archive_baseline/{exp_name}", 
                    "category": "baseline",
                    "timestamp": datetime.now().isoformat()
                })
                print(f"  ✅ {exp_name}")
            except Exception as e:
                errors_log.append(f"Failed to move {exp_name}: {e}")
                print(f"  ❌ {exp_name}: {e}")
        else:
            print(f"  ⚠️  Not found: {exp_name}")
    
    # Update consolidation log
    log_data["consolidation_actions"]["completed_archive_moves"] = moves_log
    log_data["consolidation_actions"]["archive_errors"] = errors_log
    log_data["consolidation_metadata"]["last_archive_date"] = datetime.now().isoformat()
    
    # Save updated log
    with open(base_path / "experiment_consolidation_log.json", 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n📋 Archive Summary:")
    print(f"  ✅ Successfully moved: {len(moves_log)} experiments")
    print(f"  ❌ Errors: {len(errors_log)}")
    
    if errors_log:
        print(f"\n❌ Errors encountered:")
        for error in errors_log:
            print(f"    {error}")
    
    # Show final structure
    remaining = [d for d in os.listdir(exp_path) 
                if (exp_path / d).is_dir() and not d.startswith('.')]
    
    print(f"\n📁 Final experiments/ structure:")
    for item in sorted(remaining):
        if item.startswith('archive'):
            archive_path = exp_path / item
            archive_count = len([d for d in os.listdir(archive_path) 
                               if (archive_path / d).is_dir()])
            print(f"  📦 {item}/ ({archive_count} experiments)")
        else:
            print(f"  📁 {item}/")
    
    print(f"\n🎯 Next Steps:")
    print(f"  1. Review experiment_consolidation_log.json for move documentation")
    print(f"  2. Add consolidation files to git:")
    print(f"     git add model_consolidation_report.md")
    print(f"     git add models_registry_consolidated.json") 
    print(f"     git add experiment_consolidation_log.json")
    print(f"     git add git_consolidation_strategy.md")
    print(f"  3. Commit with: git commit -m 'feat: Archive model experiments for consolidation'")
    print(f"  4. The archive/ directories contain all experiments - nothing deleted!")

def main():
    response = input("This will move experiments to archive directories (no deletion). Continue? (y/N): ")
    
    if response.lower() != 'y':
        print("Archiving cancelled.")
        return
    
    archive_experiments_safely()

if __name__ == "__main__":
    main()
