#!/usr/bin/env python3
"""
Script to preserve current model state and create baseline checkpoint.
Run this before making any architectural changes.
"""

import os
import shutil
import json
import datetime
from pathlib import Path
import git

def create_model_snapshot():
    """Create a complete snapshot of current model state."""
    
    # Create timestamp for this snapshot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_name = f"baseline_v1_{timestamp}"
    
    # Create snapshots directory
    snapshots_dir = Path("model_snapshots")
    snapshots_dir.mkdir(exist_ok=True)
    
    current_snapshot_dir = snapshots_dir / snapshot_name
    current_snapshot_dir.mkdir(exist_ok=True)
    
    print(f"Creating model snapshot: {snapshot_name}")
    
    # 1. Copy current configuration
    config_snapshot_dir = current_snapshot_dir / "config"
    config_snapshot_dir.mkdir(exist_ok=True)
    
    if Path("config").exists():
        shutil.copytree("config", config_snapshot_dir, dirs_exist_ok=True)
        print("✓ Configuration files preserved")
    
    # 2. Copy source code
    src_snapshot_dir = current_snapshot_dir / "src"
    if Path("src").exists():
        shutil.copytree("src", src_snapshot_dir, dirs_exist_ok=True)
        print("✓ Source code preserved")
    
    # 3. Copy any trained models
    models_snapshot_dir = current_snapshot_dir / "models"
    if Path("models").exists():
        shutil.copytree("models", models_snapshot_dir, dirs_exist_ok=True)
        print("✓ Model weights preserved")
    
    # 4. Copy requirements and project files
    project_files = [
        "requirements.txt", "pyproject.toml", "poetry.lock",
        "main.py", "predict.py", "README.md"
    ]
    
    for file in project_files:
        if Path(file).exists():
            shutil.copy2(file, current_snapshot_dir / file)
    
    print("✓ Project files preserved")
    
    # 5. Get git information
    try:
        repo = git.Repo(".")
        git_info = {
            "commit_hash": repo.head.commit.hexsha,
            "branch": repo.active_branch.name,
            "commit_message": repo.head.commit.message.strip(),
            "commit_date": repo.head.commit.committed_datetime.isoformat(),
            "is_dirty": repo.is_dirty()
        }
        print(f"✓ Git commit: {git_info['commit_hash'][:8]}")
    except:
        git_info = {"error": "Not a git repository or git not available"}
        print("⚠ Git information not available")
    
    # 6. Create metadata file
    metadata = {
        "snapshot_name": snapshot_name,
        "created_at": datetime.datetime.now().isoformat(),
        "description": "Baseline model - current architecture before UMLS integration",
        "model_config": {
            "n_embd": 256,
            "num_heads": 4,
            "n_layer": 3,
            "max_seq_length": 256,
            "num_classes": 3
        },
        "git_info": git_info,
        "performance_notes": "Synthetic data training - ~80-90% accuracy on sample data",
        "next_experiments": [
            "UMLS vocabulary integration",
            "Architecture scaling",
            "Real clinical data training"
        ]
    }
    
    with open(current_snapshot_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ Metadata saved")
    
    # 7. Create experiment log entry
    create_experiment_log_entry(metadata, current_snapshot_dir)
    
    print(f"\n🎯 Snapshot created successfully at: {current_snapshot_dir}")
    print(f"This preserves your current model architecture and code state.")
    
    return current_snapshot_dir

def create_experiment_log_entry(metadata, snapshot_dir):
    """Create or update experiment tracking log."""
    
    experiments_log = Path("experiments_log.jsonl")
    
    log_entry = {
        "experiment_id": metadata["snapshot_name"],
        "timestamp": metadata["created_at"],
        "model_type": "baseline_transformer",
        "architecture": metadata["model_config"],
        "description": metadata["description"],
        "snapshot_path": str(snapshot_dir),
        "status": "preserved",
        "tags": ["baseline", "transformer", "3-class", "clinical-text"]
    }
    
    # Append to JSONL file
    with open(experiments_log, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    print("✓ Experiment log updated")

if __name__ == "__main__":
    create_model_snapshot()
