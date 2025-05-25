#!/usr/bin/env python3
"""
Debug script to check config loading and types.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from utils import load_config

def debug_config():
    """Debug configuration loading and types."""
    print("=== DEBUGGING CONFIG LOADING ===")
    
    config = load_config('config/config.yaml')
    
    print("Full config:")
    print(config)
    print()
    
    print("Training config:")
    training_config = config.get('training', {})
    print(training_config)
    print()
    
    print("Learning rate:")
    lr = training_config.get('learning_rate', 2e-5)
    print(f"Value: {lr}")
    print(f"Type: {type(lr)}")
    print()
    
    print("Weight decay:")
    wd = training_config.get('weight_decay', 0.01)
    print(f"Value: {wd}")
    print(f"Type: {type(wd)}")
    print()
    
    # Try to create the optimizer manually
    import torch
    from torch.optim import AdamW
    
    # Create a dummy model to test optimizer
    dummy_model = torch.nn.Linear(10, 3)
    
    print("Testing optimizer creation:")
    try:
        optimizer = AdamW(
            dummy_model.parameters(),
            lr=lr,
            weight_decay=wd
        )
        print("✓ Optimizer created successfully")
    except Exception as e:
        print(f"❌ Optimizer creation failed: {e}")
        print(f"lr type: {type(lr)}, wd type: {type(wd)}")

if __name__ == "__main__":
    debug_config()
