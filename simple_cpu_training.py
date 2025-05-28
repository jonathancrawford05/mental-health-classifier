#!/usr/bin/env python3
"""
Simple CPU Training - Exact Replica of Your Working Setup
"""

import os
import sys
from pathlib import Path

# Simple environment setup
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from src.utils.consolidated_experiment_tracker import consolidated_tracker

def simple_cpu_training():
    """Simple CPU training replicating your successful setup."""
    
    print("🖥️  Simple CPU Training - Replicating Your Success")
    print("=" * 50)
    
    # Simple CPU setup
    torch.set_num_threads(4)
    device = torch.device('cpu')
    print(f"✅ Device: {device}")
    print(f"✅ Threads: {torch.get_num_threads()}")
    
    try:
        # Import modules
        from src.data import DataProcessor
        from src.models import create_model
        from src.training import create_trainer
        
        # Use your exact successful configuration
        config = {
            "model": {
                "name": "MentalHealthClassformer_Simple",
                "vocab_size": 10000,
                "n_embd": 256,
                "num_heads": 4,
                "n_layer": 3,
                "num_classes": 3,
                "max_seq_length": 256,
                "dropout": 0.3
            },
            "training": {
                "batch_size": 16,
                "learning_rate": 5e-05,
                "weight_decay": 0.1,
                "num_epochs": 2,  # Shorter for testing
                "warmup_steps": 100,
                "gradient_clip_norm": 1.0,
                "save_every": 1000,
                "eval_every": 200
            },
            "data": {
                "train_path": "data/train.csv",
                "val_path": "data/val.csv",
                "test_path": "data/test.csv",
                "text_column": "text",
                "label_column": "label",
                "max_length": 256
            },
            "labels": {
                "depression": 0,
                "anxiety": 1,
                "suicide": 2,
                "label_names": ["Depression", "Anxiety", "Suicide"]
            },
            "device": "cpu"
        }
        
        # Start experiment
        experiment_id = consolidated_tracker.start_experiment(
            experiment_name="simple_cpu_training",
            description="Simple CPU training replicating successful setup",
            config=config,
            tags=["cpu", "simple", "replica"]
        )
        
        print(f"✅ Started experiment: {experiment_id}")
        
        # Data processing (exactly like before)
        data_processor = DataProcessor(config['data'])
        train_texts, train_labels = data_processor.load_data('data/train.csv')
        val_texts, val_labels = data_processor.load_data('data/val.csv')
        test_texts, test_labels = data_processor.load_data('data/test.csv')
        
        print(f"✅ Data loaded: {len(train_texts)} train samples")
        
        # Build vocabulary
        data_processor.build_vocabulary(train_texts)
        config['model']['vocab_size'] = len(data_processor.vocab)
        
        print(f"✅ Vocabulary: {len(data_processor.vocab)} tokens")
        
        # Create data loaders
        dataloaders = data_processor.create_dataloaders(
            train_texts, train_labels,
            val_texts, val_labels,
            test_texts, test_labels
        )
        
        print(f"✅ Data loaders created")
        
        # Create model
        model = create_model(config['model'])
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"✅ Model created: {total_params:,} parameters")
        
        # Calculate class weights
        class_weights = data_processor.get_class_weights(train_labels)
        class_weights = class_weights.to(device)
        
        # Create trainer
        trainer = create_trainer(
            model=model,
            config=config['training'],
            device=device,
            class_weights=class_weights
        )
        
        print(f"✅ Trainer ready")
        print(f"🚀 Starting training...")
        
        # THE CRITICAL PART - this should work like before
        trainer.train(dataloaders['train'], dataloaders['val'])
        
        print(f"✅ Training completed!")
        
        # Get final metrics
        final_metrics = {
            'accuracy': trainer.history['val_acc'][-1] if trainer.history['val_acc'] else 0,
            'f1_macro': trainer.history['val_f1'][-1] if trainer.history['val_f1'] else 0,
            'best_val_f1': trainer.best_val_f1
        }
        
        # Finish experiment
        consolidated_tracker.finish_experiment(
            experiment_id,
            final_metrics=final_metrics,
            notes="Simple CPU training completed successfully"
        )
        
        print(f"🎉 Success! Results:")
        print(f"   Accuracy: {final_metrics['accuracy']:.3f}")
        print(f"   F1-Macro: {final_metrics['f1_macro']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-summary", action="store_true")
    args = parser.parse_args()
    
    if args.experiment_summary:
        summary = consolidated_tracker.get_experiment_summary()
        print(f"📊 Status: {summary['active_experiments']} active, {summary['total_experiments']} total")
        print()
    
    success = simple_cpu_training()
    
    if success:
        print(f"\n✅ Your CPU training works! The system is ready.")
    else:
        print(f"\n❌ Still having issues - may need to run on different hardware.")
