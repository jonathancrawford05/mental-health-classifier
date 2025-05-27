#!/usr/bin/env python3
"""
CPU-Only Training Script - Based on Your Working Configuration
This replicates the settings that successfully trained your baseline_v1 model
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Set CPU-only environment before any imports
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable MPS
os.environ['OMP_NUM_THREADS'] = '4'  # Use some threads but not too many
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from src.utils.consolidated_experiment_tracker import consolidated_tracker

def force_cpu_only():
    """Force CPU-only operation like your successful training."""
    # Force CPU device without MPS methods
    device = torch.device('cpu')
    
    # Set reasonable thread count
    torch.set_num_threads(4)
    
    print(f"✅ Forced CPU-only mode: {device}")
    print(f"✅ PyTorch threads: {torch.get_num_threads()}")
    
    return device

def create_working_config():
    """Create config based on your successful baseline_v1 training."""
    return {
        "model": {
            "name": "MentalHealthTransformer_CPUOptimized",
            "vocab_size": 10000,      # Same as successful training
            "n_embd": 256,            # Same as successful training
            "num_heads": 4,           # Same as successful training
            "n_layer": 3,             # Same as successful training
            "num_classes": 3,
            "max_seq_length": 256,    # Same as successful training
            "dropout": 0.3            # Same as successful training
        },
        "training": {
            "batch_size": 16,         # Reduced from 32 for stability
            "learning_rate": 5e-05,   # Same as successful training
            "weight_decay": 0.1,      # Same as successful training
            "num_epochs": 3,          # Reduced for testing
            "warmup_steps": 100,      # Reduced from 500
            "gradient_clip_norm": 1.0,
            "save_every": 1000,
            "eval_every": 200,
            "num_workers": 0          # Explicit CPU-only data loading
        },
        "data": {
            "train_path": "data/train.csv",
            "val_path": "data/val.csv",
            "test_path": "data/test.csv",
            "text_column": "text",
            "label_column": "label",
            "max_length": 256,        # Same as successful training
            "num_workers": 0          # Explicit CPU-only
        },
        "labels": {
            "depression": 0,
            "anxiety": 1,
            "suicide": 2,
            "label_names": ["Depression", "Anxiety", "Suicide"]
        },
        "paths": {
            "model_save_dir": "models/",
            "log_dir": "logs/",
            "vocab_path": "data/vocab.json"
        },
        "logging": {
            "use_wandb": false,
            "project_name": "mental-health-classifier",
            "experiment_name": "cpu-optimized",
            "log_level": "INFO"
        },
        "device": "cpu",              # Explicit CPU
        "clinical_vocab": {
            "use_umls": false,        # Disabled for stability
            "umls_api_key": null,
            "expand_synonyms": false
        }
    }

def cpu_training():
    """Run training with CPU-optimized settings matching your successful run."""
    
    print("🖥️  CPU-Optimized Training (Based on Working baseline_v1)")
    print("=" * 60)
    
    # Force CPU mode
    device = force_cpu_only()
    
    try:
        # Import training components
        from src.data import DataProcessor
        from src.models import create_model
        from src.training import create_trainer
        
        # Create working configuration
        config = create_working_config()
        config['device'] = str(device)
        
        # Start experiment tracking
        experiment_id = consolidated_tracker.start_experiment(
            experiment_name="cpu_optimized_training",
            description="CPU-optimized training based on successful baseline_v1 configuration",
            config=config,
            tags=["cpu-optimized", "baseline-replica", "production-candidate"]
        )
        
        print(f"✅ Started experiment: {experiment_id}")
        
        # Initialize data processor with CPU-safe settings
        data_config = config['data'].copy()
        data_processor = DataProcessor(data_config)
        
        print("✅ DataProcessor initialized")
        
        # Load data (same as successful training)
        train_texts, train_labels = data_processor.load_data(config['data']['train_path'])
        val_texts, val_labels = data_processor.load_data(config['data']['val_path'])
        test_texts, test_labels = data_processor.load_data(config['data']['test_path'])
        
        print(f"✅ Loaded data: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
        
        # Log data stats
        data_stats = {
            "train_samples": len(train_texts),
            "val_samples": len(val_texts),
            "test_samples": len(test_texts)
        }
        consolidated_tracker.log_metrics(experiment_id, data_stats, step=0)
        
        # Build vocabulary (same process as successful training)
        data_processor.build_vocabulary(train_texts)
        config['model']['vocab_size'] = len(data_processor.vocab)
        
        print(f"✅ Built vocabulary: {len(data_processor.vocab)} tokens")
        
        # Create data loaders with explicit CPU settings
        print("🔄 Creating CPU-safe data loaders...")
        dataloaders = data_processor.create_dataloaders(
            train_texts, train_labels,
            val_texts, val_labels,
            test_texts, test_labels
        )
        
        # Ensure data loaders use CPU-only settings
        for name, dataloader in dataloaders.items():
            if hasattr(dataloader, 'num_workers'):
                dataloader.num_workers = 0
                
        print("✅ Data loaders created with CPU-safe settings")
        
        # Create model (same architecture as successful training)
        model = create_model(config['model'])
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created: {total_params:,} parameters")
        
        # Log model stats
        model_stats = {
            "total_parameters": total_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)
        }
        consolidated_tracker.log_metrics(experiment_id, model_stats, step=0)
        
        # Calculate class weights
        class_weights = data_processor.get_class_weights(train_labels)
        class_weights = class_weights.to(device)
        
        print(f"✅ Class weights: {class_weights}")
        
        # Create trainer (same settings as successful training)
        trainer = create_trainer(
            model=model,
            config=config['training'],
            device=device,
            class_weights=class_weights
        )
        
        print("✅ Trainer created")
        
        # Start training (this is where your successful training worked)
        print("\n🚀 Starting training with proven configuration...")
        trainer.train(dataloaders['train'], dataloaders['val'])
        
        print("✅ Training completed successfully!")
        
        # Save model
        consolidated_tracker.save_model(
            experiment_id,
            model,
            "cpu_optimized_model",
            additional_info={
                "vocab_size": len(data_processor.vocab),
                "class_weights": class_weights.cpu().tolist(),
                "device": str(device)
            }
        )
        
        # Evaluate on test set
        print("📊 Running test evaluation...")
        test_metrics = trainer.evaluate(
            dataloaders['test'],
            save_plots=True,
            save_dir=f'experiments/{experiment_id}/results/'
        )
        
        # Log final metrics
        consolidated_tracker.log_metrics(experiment_id, test_metrics, step=-1)
        
        # Finish experiment
        final_metrics = {
            'loss': trainer.history['val_loss'][-1] if trainer.history['val_loss'] else 0,
            'accuracy': trainer.history['val_acc'][-1] if trainer.history['val_acc'] else 0,
            'f1_macro': trainer.history['val_f1'][-1] if trainer.history['val_f1'] else 0,
            'best_val_f1': trainer.best_val_f1
        }
        
        consolidated_tracker.finish_experiment(
            experiment_id,
            final_metrics=final_metrics,
            notes="CPU-optimized training completed successfully using proven configuration"
        )
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 Final Results:")
        print(f"   Accuracy: {final_metrics['accuracy']:.3f}")
        print(f"   F1-Macro: {final_metrics['f1_macro']:.3f}")
        print(f"   Best F1: {final_metrics['best_val_f1']:.3f}")
        
        # Check production promotion
        if final_metrics['accuracy'] >= 0.55 and final_metrics['f1_macro'] >= 0.45:
            print(f"\n🏆 This model qualifies for production promotion!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="CPU-optimized training")
    parser.add_argument("--experiment-summary", action="store_true", help="Show experiment summary")
    args = parser.parse_args()
    
    if args.experiment_summary:
        summary = consolidated_tracker.get_experiment_summary()
        print("📊 Current Experiment Status:")
        print("=" * 30)
        print(f"Active: {summary['active_experiments']}")
        print(f"Production: {summary['production_models']}")
        print(f"Total: {summary['total_experiments']}")
        print("=" * 30)
    
    success = cpu_training()
    
    if success:
        print(f"\n✅ CPU training successful!")
        print(f"Your system can handle training with the right configuration.")
    else:
        print(f"\n❌ Training failed despite CPU optimization.")

if __name__ == "__main__":
    main()
