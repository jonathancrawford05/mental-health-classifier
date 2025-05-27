#!/usr/bin/env python3
"""
Medium-Scale CPU Training Experiment
Using all learnings to date for optimal architecture and training
"""

import os
import sys
from pathlib import Path

# Environment setup
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import json
import shutil
from datetime import datetime

def medium_cpu_experiment():
    """Medium-scale CPU experiment with optimized architecture."""
    
    print("🚀 MEDIUM-SCALE CPU TRAINING EXPERIMENT")
    print("=" * 60)
    print("Using learnings from successful minimal training")
    print("Scaling up architecture while maintaining CPU efficiency")
    print("=" * 60)
    
    # Optimal CPU setup based on learnings
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
        from src.utils.consolidated_experiment_tracker import consolidated_tracker
        
        # MEDIUM EXPERIMENT CONFIGURATION
        # Based on learnings: larger model, more data, longer training
        config = {
            "model": {
                "name": "MentalHealthTransformer_Medium",
                "vocab_size": 15000,  # Larger vocab for better representation
                "n_embd": 384,        # Increased from 256 (50% larger)
                "num_heads": 6,       # Increased from 4 (more attention heads)
                "n_layer": 4,         # Increased from 3 (deeper network)
                "num_classes": 3,
                "max_seq_length": 384, # Longer sequences for context
                "dropout": 0.2        # Reduced dropout (less regularization for larger model)
            },
            "training": {
                "batch_size": 24,     # Larger batches (still CPU-friendly)
                "learning_rate": 3e-05, # Slightly lower LR for stability
                "weight_decay": 0.05,  # Reduced weight decay
                "num_epochs": 8,      # More epochs for convergence
                "warmup_steps": 200,  # More warmup steps
                "gradient_clip_norm": 1.0,
                "save_every": 500,
                "eval_every": 100     # More frequent evaluation
            },
            "data": {
                "train_path": "data/train.csv",
                "val_path": "data/val.csv", 
                "test_path": "data/test.csv",
                "text_column": "text",
                "label_column": "label",
                "max_length": 384
            },
            "labels": {
                "depression": 0,
                "anxiety": 1,
                "suicide": 2,
                "label_names": ["Depression", "Anxiety", "Suicide"]
            },
            "device": "cpu"
        }
        
        # Start experiment with tracking
        experiment_id = consolidated_tracker.start_experiment(
            experiment_name="medium_cpu_experiment",
            description="Medium-scale CPU training with optimized architecture based on successful minimal training",
            config=config,
            tags=["cpu", "medium", "optimized", "production"]
        )
        
        print(f"🧪 Started experiment: {experiment_id}")
        
        # Setup directories
        exp_dir = Path("experiments") / experiment_id
        models_dir = exp_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced data processing
        print(f"\n📊 Data Processing:")
        data_processor = DataProcessor(config['data'])
        
        # Check if we need larger dataset
        train_texts, train_labels = data_processor.load_data('data/train.csv')
        val_texts, val_labels = data_processor.load_data('data/val.csv')
        test_texts, test_labels = data_processor.load_data('data/test.csv')
        
        print(f"   Training samples: {len(train_texts)}")
        print(f"   Validation samples: {len(val_texts)}")
        print(f"   Test samples: {len(test_texts) if test_texts else 0}")
        
        # Create larger dataset if current is too small
        if len(train_texts) < 2000:
            print(f"   📈 Current dataset small - creating larger sample...")
            from src.data import create_sample_data
            
            # Create larger sample dataset
            create_sample_data("data/medium_train.csv", 2500)
            
            # Re-split for medium experiment
            import pandas as pd
            from sklearn.model_selection import train_test_split
            
            df = pd.read_csv("data/medium_train.csv")
            
            # Better split ratios for medium experiment
            train_val_df, test_df = train_test_split(
                df, test_size=0.15, random_state=42, stratify=df['label']
            )
            
            train_df, val_df = train_test_split(
                train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['label']
            )
            
            # Save new splits
            train_df.to_csv("data/train.csv", index=False)
            val_df.to_csv("data/val.csv", index=False) 
            test_df.to_csv("data/test.csv", index=False)
            
            # Reload data
            train_texts, train_labels = data_processor.load_data('data/train.csv')
            val_texts, val_labels = data_processor.load_data('data/val.csv')
            test_texts, test_labels = data_processor.load_data('data/test.csv')
            
            print(f"   ✅ Enhanced dataset created:")
            print(f"      Training: {len(train_texts)} samples")
            print(f"      Validation: {len(val_texts)} samples")
            print(f"      Test: {len(test_texts)} samples")
        
        # Build vocabulary
        data_processor.build_vocabulary(train_texts)
        actual_vocab_size = len(data_processor.vocab)
        config['model']['vocab_size'] = actual_vocab_size
        
        print(f"   ✅ Vocabulary: {actual_vocab_size} tokens (target: {config['model']['vocab_size']})")
        
        # Create data loaders
        dataloaders = data_processor.create_dataloaders(
            train_texts, train_labels,
            val_texts, val_labels,
            test_texts, test_labels
        )
        
        print(f"   ✅ Data loaders created")
        
        # Create medium-scale model
        print(f"\n🧠 Model Architecture:")
        model = create_model(config['model']).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: ~{total_params * 4 / (1024**2):.1f} MB")
        print(f"   Architecture: {config['model']['n_layer']} layers, {config['model']['n_embd']} dim, {config['model']['num_heads']} heads")
        
        # Calculate class weights
        class_weights = data_processor.get_class_weights(train_labels)
        class_weights = class_weights.to(device)
        print(f"   Class weights: Depression={class_weights[0]:.2f}, Anxiety={class_weights[1]:.2f}, Suicide={class_weights[2]:.2f}")
        
        # Setup trainer with experiment directory
        trainer_config = config['training'].copy()
        trainer_config['model_save_dir'] = str(models_dir)
        
        trainer = create_trainer(
            model=model,
            config=trainer_config,
            device=device,
            class_weights=class_weights
        )
        
        print(f"   ✅ Trainer configured")
        
        # Training
        print(f"\n🏋️ Training Phase:")
        print(f"   Epochs: {config['training']['num_epochs']}")
        print(f"   Batch size: {config['training']['batch_size']}")
        print(f"   Learning rate: {config['training']['learning_rate']}")
        print(f"   Expected duration: ~{config['training']['num_epochs'] * 3:.0f} minutes")
        
        start_time = datetime.now()
        trainer.train(dataloaders['train'], dataloaders['val'])
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        print(f"   ✅ Training completed in {training_duration/60:.1f} minutes")
        
        # Save all model components
        print(f"\n💾 Saving Model Components:")
        
        # 1. Best model checkpoint
        best_model_path = models_dir / "best_model.pt"
        checkpoint = {
            'epoch': len(trainer.history['val_f1']),
            'model_state_dict': trainer.best_model_state or trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'best_val_f1': trainer.best_val_f1,
            'config': config,
            'vocab_size': len(data_processor.vocab),
            'label_names': config['labels']['label_names'],
            'training_duration_minutes': training_duration/60,
            'total_parameters': total_params
        }
        torch.save(checkpoint, best_model_path)
        print(f"   ✅ Model checkpoint: {best_model_path}")
        
        # 2. Vocabulary
        vocab_path = models_dir / "vocab.pkl"
        data_processor.save_vocabulary(str(vocab_path))
        print(f"   ✅ Vocabulary: {vocab_path}")
        
        # 3. Model info
        model_info_path = models_dir / "model_info.json"
        save_model_info(model, config, str(model_info_path))
        print(f"   ✅ Model info: {model_info_path}")
        
        # 4. Training history
        history_path = models_dir / "training_history.json"
        trainer.save_training_history(str(history_path))
        print(f"   ✅ Training history: {history_path}")
        
        # 5. Copy to main models directory for easy access
        main_models_dir = Path("models")
        main_models_dir.mkdir(exist_ok=True)
        
        shutil.copy2(best_model_path, main_models_dir / "best_model.pt")
        shutil.copy2(vocab_path, main_models_dir / "vocab.pkl")
        shutil.copy2(model_info_path, main_models_dir / "model_info.json")
        print(f"   ✅ Models copied to main directory")
        
        # Final evaluation and metrics
        print(f"\n📊 Results Summary:")
        
        final_metrics = {
            'accuracy': trainer.history['val_acc'][-1] if trainer.history['val_acc'] else 0,
            'f1_macro': trainer.history['val_f1'][-1] if trainer.history['val_f1'] else 0,
            'best_val_f1': trainer.best_val_f1,
            'total_parameters': total_params,
            'vocab_size': len(data_processor.vocab),
            'training_duration_minutes': training_duration/60,
            'training_samples': len(train_texts),
            'model_size_mb': total_params * 4 / (1024**2)
        }
        
        # Test evaluation if available
        if test_texts:
            print(f"   🧪 Evaluating on test set...")
            test_metrics = trainer.evaluate(dataloaders['test'])
            final_metrics.update({
                'test_accuracy': test_metrics['accuracy'],
                'test_f1_macro': test_metrics['f1_macro'],
                'test_precision_macro': test_metrics['precision_macro'],
                'test_recall_macro': test_metrics['recall_macro']
            })
        
        # Log metrics and finish experiment
        consolidated_tracker.log_metrics(experiment_id, final_metrics)
        consolidated_tracker.save_model(experiment_id, model, "medium_model", additional_info=final_metrics)
        consolidated_tracker.finish_experiment(experiment_id, final_metrics=final_metrics)
        
        # Print comprehensive results
        print(f"\n🎉 MEDIUM EXPERIMENT COMPLETED!")
        print(f"=" * 60)
        print(f"Experiment ID: {experiment_id}")
        print(f"Model Architecture: {config['model']['n_layer']}L-{config['model']['n_embd']}D-{config['model']['num_heads']}H")
        print(f"Parameters: {total_params:,} ({total_params * 4 / (1024**2):.1f} MB)")
        print(f"Training Duration: {training_duration/60:.1f} minutes")
        print(f"Training Samples: {len(train_texts):,}")
        print(f"")
        print(f"📈 Performance Metrics:")
        print(f"   Best Validation F1: {final_metrics['best_val_f1']:.4f}")
        print(f"   Final Validation Accuracy: {final_metrics['accuracy']:.4f}")
        if 'test_f1_macro' in final_metrics:
            print(f"   Test F1 Macro: {final_metrics['test_f1_macro']:.4f}")
            print(f"   Test Accuracy: {final_metrics['test_accuracy']:.4f}")
        print(f"")
        print(f"🎯 Next Steps:")
        print(f"   • Test model: python predict.py --text 'your text here'")
        print(f"   • Interactive mode: python predict.py --interactive")
        print(f"   • Compare with minimal model performance")
        print(f"=" * 60)
        
        return True, experiment_id, final_metrics
        
    except Exception as e:
        print(f"❌ Medium experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

if __name__ == "__main__":
    success, exp_id, metrics = medium_cpu_experiment()
    
    if success:
        print(f"\n✅ Medium experiment completed successfully!")
        print(f"Experiment ID: {exp_id}")
        
        # Quick test of the new model
        print(f"\n🔮 Quick test of medium model...")
        try:
            from predict import MentalHealthPredictor
            predictor = MentalHealthPredictor()
            predictor.load_model()
            
            test_text = "I've been feeling overwhelmed and hopeless lately"
            prediction, probs = predictor.predict(test_text)
            
            print(f"Test: '{test_text}'")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {max(probs.values()):.3f}")
            
            print(f"\n🎉 Medium model is working!")
            
        except Exception as e:
            print(f"⚠️ Model test failed: {e}")
            
    else:
        print(f"\n❌ Medium experiment failed")
