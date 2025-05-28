#!/usr/bin/env python3
"""
Optimized Configuration for Mental Health Classifier

This creates a smaller, more efficient model architecture
better suited for the current dataset size (196 vocab tokens).
"""

import json
from pathlib import Path

def create_optimized_config():
    """Create optimized configuration for smaller, more efficient model."""
    
    # Optimized for small vocabulary and CPU training
    config = {
        "model": {
            "name": "MentalHealthTransformer_Optimized",
            "vocab_size": 500,        # Reasonable size for expansion
            "n_embd": 256,           # Reduced from 512 - still effective
            "num_heads": 4,          # Reduced from 8 - sufficient for small data
            "n_layer": 4,            # Reduced from 6 - prevents overfitting
            "num_classes": 3,
            "max_seq_length": 256,   # Reduced from 512 - faster training
            "dropout": 0.1           # Standard dropout
        },
        "training": {
            "batch_size": 16,        # Smaller batches for CPU
            "learning_rate": 1e-4,   # Higher LR for smaller model
            "weight_decay": 0.01,    # Lighter regularization
            "num_epochs": 15,        # More epochs for smaller model
            "warmup_steps": 200,     # Proportional warmup
            "gradient_clip_norm": 1.0,
            "save_every": 100,
            "eval_every": 25,        # More frequent evaluation
            "early_stopping": {
                "patience": 5,       # More patience
                "min_delta": 0.001,
                "performance_threshold": 0.85  # Realistic target
            }
        },
        "data": {
            "train_path": "data/train.csv",
            "val_path": "data/val.csv", 
            "test_path": "data/test.csv",
            "text_column": "text",
            "label_column": "label",
            "max_length": 256        # Consistent with model
        },
        "labels": {
            "depression": 0,
            "anxiety": 1,
            "suicide": 2,
            "label_names": ["Depression", "Anxiety", "Suicide"]
        },
        "device": "cpu",
        "paths": {
            "model_save_dir": "models/optimized",
            "results_dir": "results/optimized"
        },
        "logging": {
            "log_level": "INFO",
            "log_file": "logs/optimized_training.log"
        }
    }
    
    # Calculate expected model size
    vocab_size = config['model']['vocab_size']
    n_embd = config['model']['n_embd']
    n_layer = config['model']['n_layer']
    num_heads = config['model']['num_heads']
    
    # Rough parameter calculation
    embedding_params = vocab_size * n_embd * 2  # token + position
    attention_params = n_layer * (4 * n_embd * n_embd)  # Q,K,V,O projections
    mlp_params = n_layer * (n_embd * n_embd * 8)  # 4x expansion, 2 layers
    classifier_params = n_embd * 3  # final classifier
    
    total_params = embedding_params + attention_params + mlp_params + classifier_params
    model_size_mb = total_params * 4 / (1024**2)
    
    config['model']['estimated_parameters'] = total_params
    config['model']['estimated_size_mb'] = model_size_mb
    
    return config

def save_optimized_config(config_path="config/optimized_config.yaml"):
    """Save optimized configuration to file."""
    config = create_optimized_config()
    
    # Create config directory if it doesn't exist
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save as YAML
    try:
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"✅ Optimized config saved to: {config_path}")
    except ImportError:
        # Fallback to JSON if PyYAML not available
        json_path = config_path.replace('.yaml', '.json')
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Optimized config saved to: {json_path}")
    
    return config

def print_config_comparison():
    """Print comparison between current and optimized configurations."""
    
    # Load current config
    try:
        with open("models/model_info.json", 'r') as f:
            current_info = json.load(f)
            current_config = current_info['config']['model']
            current_params = current_info['total_parameters']
    except:
        print("❌ Could not load current model config")
        return
    
    # Get optimized config
    optimized = create_optimized_config()['model']
    
    print("📊 MODEL ARCHITECTURE COMPARISON")
    print("=" * 50)
    print(f"{'Metric':<20} {'Current':<15} {'Optimized':<15} {'Change'}")
    print("-" * 60)
    
    metrics = [
        ('Parameters', f"{current_params:,}", f"{optimized['estimated_parameters']:,}"),
        ('Size (MB)', f"{current_params*4/(1024**2):.1f}", f"{optimized['estimated_size_mb']:.1f}"),
        ('Embedding Dim', current_config['n_embd'], optimized['n_embd']),
        ('Layers', current_config['n_layer'], optimized['n_layer']),
        ('Attention Heads', current_config['num_heads'], optimized['num_heads']),
        ('Max Length', current_config['max_seq_length'], optimized['max_seq_length'])
    ]
    
    for metric, current, optimized_val in metrics:
        if isinstance(current, str):
            print(f"{metric:<20} {current:<15} {optimized_val:<15}")
        else:
            change = "↓" if optimized_val < current else "↑" if optimized_val > current else "="
            print(f"{metric:<20} {current:<15} {optimized_val:<15} {change}")
    
    # Calculate parameter reduction
    param_reduction = (current_params - optimized['estimated_parameters']) / current_params * 100
    print("\n🎯 OPTIMIZATION BENEFITS:")
    print(f"   • {param_reduction:.1f}% parameter reduction")
    print(f"   • ~{param_reduction/2:.1f}% faster training")
    print(f"   • Better suited for small datasets")
    print(f"   • Reduced overfitting risk")
    print(f"   • More stable CPU training")

if __name__ == "__main__":
    print("🚀 Creating Optimized Configuration...")
    
    # Show comparison
    print_config_comparison()
    
    # Save optimized config
    config = save_optimized_config()
    
    print(f"\n✨ OPTIMIZED MODEL ARCHITECTURE:")
    print(f"   • {config['model']['estimated_parameters']:,} parameters")
    print(f"   • {config['model']['estimated_size_mb']:.1f} MB model size")
    print(f"   • {config['model']['n_layer']} layers × {config['model']['n_embd']} dimensions")
    print(f"   • {config['model']['num_heads']} attention heads")
    print(f"   • Max sequence length: {config['model']['max_seq_length']}")
    print(f"\n🎯 Use with: python optimized_training.py")
