#!/usr/bin/env python3
"""
Fixed CPU Training - Complete Model Saving and Integration
"""

import os
import sys
from pathlib import Path

# Simple environment setup
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import json
from src.utils.consolidated_experiment_tracker import consolidated_tracker

def fixed_cpu_training():
    """Fixed CPU training with proper model saving and integration."""
    
    print("🖥️  Fixed CPU Training - Complete Integration")
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
        from src.utils import save_model_info
        
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
                "num_epochs": 2,
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
            experiment_name="fixed_cpu_training",
            description="Fixed CPU training with complete model saving",
            config=config,
            tags=["cpu", "fixed", "production"]
        )
        
        print(f"✅ Started experiment: {experiment_id}")
        
        # Setup experiment directories
        exp_dir = Path("experiments") / experiment_id
        models_dir = exp_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # Update config for trainer to save in experiment directory
        trainer_config = config['training'].copy()
        trainer_config['model_save_dir'] = str(models_dir)
        
        # Create trainer
        trainer = create_trainer(
            model=model,
            config=trainer_config,
            device=device,
            class_weights=class_weights
        )
        
        print(f"✅ Trainer ready")
        print(f"🚀 Starting training...")
        
        # Training
        trainer.train(dataloaders['train'], dataloaders['val'])
        
        print(f"✅ Training completed!")
        
        # *** CRITICAL FIX: Explicit Model Saving ***
        
        # 1. Save best model checkpoint
        best_model_path = models_dir / "best_model.pt"
        checkpoint = {
            'epoch': len(trainer.history['val_f1']),
            'model_state_dict': trainer.best_model_state or trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'best_val_f1': trainer.best_val_f1,
            'config': config,
            'vocab_size': len(data_processor.vocab),
            'label_names': config['labels']['label_names']
        }
        torch.save(checkpoint, best_model_path)
        print(f"💾 Model saved: {best_model_path}")
        
        # 2. Save vocabulary
        vocab_path = models_dir / "vocab.pkl"
        data_processor.save_vocabulary(str(vocab_path))
        print(f"💾 Vocabulary saved: {vocab_path}")
        
        # 3. Save model info for inference
        model_info_path = models_dir / "model_info.json"
        save_model_info(model, config, str(model_info_path))
        print(f"💾 Model info saved: {model_info_path}")
        
        # 4. Copy to main models directory for easy access
        main_models_dir = Path("models")
        main_models_dir.mkdir(exist_ok=True)
        
        import shutil
        shutil.copy2(best_model_path, main_models_dir / "best_model.pt")
        shutil.copy2(vocab_path, main_models_dir / "vocab.pkl")
        shutil.copy2(model_info_path, main_models_dir / "model_info.json")
        print(f"💾 Models copied to main directory: {main_models_dir}")
        
        # 5. Save training history
        history_path = models_dir / "training_history.json"
        trainer.save_training_history(str(history_path))
        print(f"💾 Training history saved: {history_path}")
        
        # Get final metrics
        final_metrics = {
            'accuracy': trainer.history['val_acc'][-1] if trainer.history['val_acc'] else 0,
            'f1_macro': trainer.history['val_f1'][-1] if trainer.history['val_f1'] else 0,
            'best_val_f1': trainer.best_val_f1,
            'total_parameters': total_params,
            'vocab_size': len(data_processor.vocab)
        }
        
        # Log final metrics with tracker
        consolidated_tracker.log_metrics(
            experiment_id, 
            final_metrics, 
            epoch=len(trainer.history['val_f1'])
        )
        
        # Save model with tracker
        consolidated_tracker.save_model(
            experiment_id,
            model,
            "best_model",
            additional_info={
                'best_val_f1': trainer.best_val_f1,
                'config': config,
                'vocab_size': len(data_processor.vocab)
            }
        )
        
        # Finish experiment
        consolidated_tracker.finish_experiment(
            experiment_id,
            final_metrics=final_metrics,
            notes="Fixed CPU training with complete model saving"
        )
        
        print(f"\n🎉 SUCCESS! Complete Results:")
        print(f"   Experiment ID: {experiment_id}")
        print(f"   Model Parameters: {total_params:,}")
        print(f"   Vocabulary Size: {len(data_processor.vocab)}")
        print(f"   Accuracy: {final_metrics['accuracy']:.3f}")
        print(f"   F1-Macro: {final_metrics['f1_macro']:.3f}")
        print(f"   Best Val F1: {final_metrics['best_val_f1']:.3f}")
        print(f"\n📁 Model files saved:")
        print(f"   • {best_model_path}")
        print(f"   • {vocab_path}")
        print(f"   • {model_info_path}")
        print(f"   • {main_models_dir}/best_model.pt")
        print(f"   • {main_models_dir}/vocab.pkl")
        print(f"   • {main_models_dir}/model_info.json")
        
        return True, experiment_id, final_metrics
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def make_cpu_training_default():
    """Make CPU training the default by updating main configuration."""
    
    print("\n🔧 Making CPU Training the Default...")
    
    # Update main config
    config_path = Path("config/config.yaml")
    
    if config_path.exists():
        import yaml
        
        # Read current config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update for CPU-first approach
        config['device'] = 'cpu'
        config['training']['batch_size'] = 16  # CPU-optimized
        config['training']['learning_rate'] = 5e-05  # Proven working
        config['training']['num_epochs'] = 5  # Reasonable for CPU
        config['model']['n_embd'] = 256  # CPU-friendly size
        config['model']['n_layer'] = 3
        config['model']['num_heads'] = 4
        
        # Add CPU training profile
        if 'profiles' not in config:
            config['profiles'] = {}
        
        config['profiles']['cpu_default'] = {
            'device': 'cpu',
            'model': {
                'n_embd': 256,
                'n_layer': 3,
                'num_heads': 4,
                'dropout': 0.3
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 5e-05,
                'num_epochs': 5,
                'weight_decay': 0.1
            }
        }
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Updated {config_path} for CPU-first training")
    
    # Update main.py to use CPU by default
    main_py_path = Path("main.py")
    if main_py_path.exists():
        with open(main_py_path, 'r') as f:
            content = f.read()
        
        # Replace default device
        updated_content = content.replace(
            'default="auto"',
            'default="cpu"'
        )
        
        with open(main_py_path, 'w') as f:
            f.write(updated_content)
        
        print(f"✅ Updated {main_py_path} to default to CPU")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-summary", action="store_true")
    parser.add_argument("--make-default", action="store_true", help="Make CPU training the default")
    args = parser.parse_args()
    
    if args.experiment_summary:
        summary = consolidated_tracker.get_experiment_summary()
        print(f"📊 Status: {summary['active_experiments']} active, {summary['total_experiments']} total")
        print()
    
    success, experiment_id, metrics = fixed_cpu_training()
    
    if success:
        print(f"\n✅ Fixed CPU training completed successfully!")
        
        if args.make_default:
            make_cpu_training_default()
            print(f"\n🎯 CPU training is now the default configuration!")
        
        print(f"\n🧪 Test the trained model:")
        print(f"   python predict.py")
        print(f"   python predict.py --interactive")
        print(f"   python predict.py --text 'I feel hopeless and sad'")
        
    else:
        print(f"\n❌ Training failed - check error messages above")
