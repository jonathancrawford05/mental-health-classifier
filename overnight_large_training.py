#!/usr/bin/env python3
"""
Overnight Large Dataset CPU Training
Designed to run while you sleep - larger dataset, robust early stopping
Expected duration: 4-6 hours for production-ready model
"""

import os
import sys
from pathlib import Path
import time
from datetime import datetime

# Environment setup
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import json
import shutil
from typing import Dict

def overnight_large_training():
    """Overnight training on large dataset - designed to run while you sleep."""
    
    print("🌙 OVERNIGHT LARGE DATASET TRAINING")
    print("=" * 70)
    print("Designed for overnight execution while you sleep")
    print("Target: 10K+ samples, production-ready model in 4-6 hours")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Robust CPU setup for long runs
    torch.set_num_threads(6)
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
        
        # OVERNIGHT LARGE DATASET CONFIGURATION
        config = {
            "model": {
                "name": "MentalHealthTransformer_Overnight",
                "vocab_size": 25000,  # Larger vocabulary
                "n_embd": 512,        # Production size
                "num_heads": 8,       # Standard
                "n_layer": 6,         # Deep enough for complexity
                "num_classes": 3,
                "max_seq_length": 512,
                "dropout": 0.1        # Lower dropout for larger dataset
            },
            "training": {
                "batch_size": 40,     # Larger batches for efficiency
                "learning_rate": 1.5e-05,  # Lower LR for stability
                "weight_decay": 0.02,
                "num_epochs": 20,     # Allow longer training
                "warmup_steps": 500,  # More warmup for large dataset
                "gradient_clip_norm": 1.0,
                "save_every": 200,
                "eval_every": 100     # Monitor frequently
            },
            "data": {
                "train_path": "data/train.csv",
                "val_path": "data/val.csv",
                "test_path": "data/test.csv",
                "text_column": "text",
                "label_column": "label",
                "max_length": 512
            },
            "labels": {
                "depression": 0,
                "anxiety": 1,
                "suicide": 2,
                "label_names": ["Depression", "Anxiety", "Suicide"]
            },
            "device": "cpu",
            "early_stopping": {
                "patience": 4,      # More patience for large dataset
                "min_delta": 0.0005,  # Smaller delta for fine improvements
                "performance_threshold": 0.98,  # High but achievable
                "max_training_hours": 6.0  # 6 hour limit (safe for overnight)
            }
        }
        
        # Start experiment
        experiment_id = consolidated_tracker.start_experiment(
            experiment_name="overnight_large_dataset",
            description=f"Overnight large dataset training started at {datetime.now().strftime('%H:%M')} - designed to complete while sleeping",
            config=config,
            tags=["cpu", "overnight", "large-dataset", "production"]
        )
        
        print(f"🧪 Started experiment: {experiment_id}")
        
        # Setup directories
        exp_dir = Path("experiments") / experiment_id
        models_dir = exp_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # CREATE LARGE DATASET (10K samples)
        print(f"\n📊 Creating Large Dataset:")
        data_processor = DataProcessor(config['data'])
        
        from src.data import create_sample_data
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        print(f"   🔄 Generating 10,000 samples for overnight training...")
        create_sample_data("data/overnight_large.csv", 10000)
        
        # Optimal splits for large dataset
        df = pd.read_csv("data/overnight_large.csv")
        
        # 75/15/10 split for large dataset
        train_val_df, test_df = train_test_split(
            df, test_size=0.10, random_state=42, stratify=df['label']
        )
        
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.167, random_state=42, stratify=train_val_df['label']  # ~15% of total
        )
        
        # Save large dataset
        train_df.to_csv("data/train.csv", index=False)
        val_df.to_csv("data/val.csv", index=False)
        test_df.to_csv("data/test.csv", index=False)
        
        # Load large dataset
        train_texts, train_labels = data_processor.load_data('data/train.csv')
        val_texts, val_labels = data_processor.load_data('data/val.csv')
        test_texts, test_labels = data_processor.load_data('data/test.csv')
        
        print(f"   ✅ Large dataset created:")
        print(f"      Training: {len(train_texts):,} samples")
        print(f"      Validation: {len(val_texts):,} samples")
        print(f"      Test: {len(test_texts):,} samples")
        print(f"      Total: {len(train_texts) + len(val_texts) + len(test_texts):,} samples")
        
        # Build large vocabulary
        data_processor.build_vocabulary(train_texts)
        actual_vocab_size = len(data_processor.vocab)
        config['model']['vocab_size'] = actual_vocab_size
        
        print(f"   ✅ Large vocabulary: {actual_vocab_size:,} tokens")
        
        # Create data loaders
        dataloaders = data_processor.create_dataloaders(
            train_texts, train_labels,
            val_texts, val_labels,
            test_texts, test_labels
        )
        
        # Create large model
        print(f"\n🏗️ Large Model Architecture:")
        model = create_model(config['model']).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / (1024**2)
        
        print(f"   Parameters: {total_params:,} ({model_size_mb:.1f} MB)")
        print(f"   Architecture: {config['model']['n_layer']}L-{config['model']['n_embd']}D-{config['model']['num_heads']}H")
        print(f"   Training samples: {len(train_texts):,}")
        print(f"   Batches per epoch: {len(dataloaders['train'])}")
        
        # Estimate training time
        estimated_epoch_time = len(dataloaders['train']) * 0.8 / 60  # minutes
        estimated_total_time = estimated_epoch_time * 8  # assume ~8 epochs
        print(f"   Est. epoch time: {estimated_epoch_time:.1f} minutes")
        print(f"   Est. total time: {estimated_total_time/60:.1f} hours")
        
        # Setup early stopping for overnight run
        from optimized_cpu_training import EarlyStopping
        early_stopping = EarlyStopping(
            patience=config['early_stopping']['patience'],
            min_delta=config['early_stopping']['min_delta'],
            performance_threshold=config['early_stopping']['performance_threshold'],
            max_training_hours=config['early_stopping']['max_training_hours']
        )
        
        # Calculate class weights
        class_weights = data_processor.get_class_weights(train_labels)
        class_weights = class_weights.to(device)
        
        # Setup trainer
        trainer_config = config['training'].copy()
        trainer_config['model_save_dir'] = str(models_dir)
        
        trainer = create_trainer(
            model=model,
            config=trainer_config,
            device=device,
            class_weights=class_weights
        )
        
        # OVERNIGHT TRAINING LOOP
        print(f"\n🌙 Starting Overnight Training:")
        print(f"   Dataset: {len(train_texts):,} samples")
        print(f"   Max duration: {config['early_stopping']['max_training_hours']} hours")
        print(f"   Early stopping: F1 > {config['early_stopping']['performance_threshold']}")
        print(f"   Patience: {config['early_stopping']['patience']} epochs")
        print(f"   Expected completion: {(datetime.now().timestamp() + estimated_total_time*3600).__format__('%H:%M')} (tomorrow morning)")
        
        early_stopping.start_training()
        start_time = datetime.now()
        
        num_epochs = config['training']['num_epochs']
        
        # Enhanced overnight training loop with reduced output
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\n{'='*50}")
            print(f"🌙 EPOCH {epoch+1}/{num_epochs} - {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*50}")
            
            # Training phase with minimal output for overnight run
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, batch in enumerate(dataloaders['train']):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                trainer.optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = trainer.criterion(outputs['logits'], labels)
                loss.backward()
                
                if trainer.config.get('gradient_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.config['gradient_clip_norm'])
                
                trainer.optimizer.step()
                if trainer.scheduler:
                    trainer.scheduler.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # Minimal progress updates for overnight (every 50 batches)
                if (batch_idx + 1) % 50 == 0:
                    elapsed = time.time() - epoch_start
                    progress_pct = ((batch_idx + 1) / len(dataloaders['train'])) * 100
                    eta_minutes = ((len(dataloaders['train']) - batch_idx - 1) * elapsed / (batch_idx + 1)) / 60
                    
                    print(f"    Batch {batch_idx+1:3d}/{len(dataloaders['train']):3d} | "
                          f"{progress_pct:5.1f}% | Loss: {loss.item():.4f} | ETA: {eta_minutes:.1f}m")
            
            avg_train_loss = train_loss / train_batches
            
            # Validation phase
            print(f"   Validating...")
            val_metrics = trainer.evaluate(dataloaders['val'])
            val_f1 = val_metrics['f1_macro']
            val_acc = val_metrics['accuracy']
            
            # Update trainer history
            trainer.history['train_loss'].append(avg_train_loss)
            trainer.history['val_loss'].append(val_metrics['loss'])
            trainer.history['val_acc'].append(val_acc)
            trainer.history['val_f1'].append(val_f1)
            
            # Update best model
            is_best = False
            if val_f1 > trainer.best_val_f1:
                trainer.best_val_f1 = val_f1
                trainer.best_model_state = model.state_dict().copy()
                is_best = True
                
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_f1': trainer.best_val_f1,
                    'config': config
                }, models_dir / 'best_model.pt')
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            elapsed_total = (datetime.now() - start_time).total_seconds() / 3600
            
            print(f"   Training Loss: {avg_train_loss:.4f}")
            print(f"   Val F1: {val_f1:.4f} {' 🌟 NEW BEST!' if is_best else ''}")
            print(f"   Val Accuracy: {val_acc:.4f}")
            print(f"   Best F1: {trainer.best_val_f1:.4f}")
            print(f"   Epoch Time: {epoch_time/60:.1f} min")
            print(f"   Total Time: {elapsed_total:.1f}h")
            print(f"   Est. Completion: {(datetime.now().timestamp() + (estimated_total_time - elapsed_total)*3600).__format__('%H:%M')}")
            
            # Log metrics
            consolidated_tracker.log_metrics(experiment_id, {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_f1_macro': val_f1,
                'val_accuracy': val_acc,
                'best_val_f1': trainer.best_val_f1,
                'epoch_time_minutes': epoch_time / 60,
                'total_time_hours': elapsed_total
            }, epoch=epoch+1)
            
            # Early stopping check
            should_stop = early_stopping(epoch + 1, val_f1, model.state_dict())
            if should_stop:
                print(f"\n🛑 Early stopping triggered: {early_stopping.stop_reason}")
                break
        
        # Training completed
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        print(f"\n✅ OVERNIGHT TRAINING COMPLETED!")
        print(f"   Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Total time: {total_duration/3600:.1f} hours")
        print(f"   Epochs: {epoch + 1}/{num_epochs}")
        print(f"   Best F1: {trainer.best_val_f1:.4f}")
        
        # Save everything
        print(f"\n💾 Saving Large Model...")
        
        # Complete model save
        best_model_path = models_dir / "best_model.pt"
        checkpoint = {
            'model_state_dict': trainer.best_model_state or model.state_dict(),
            'config': config,
            'best_val_f1': trainer.best_val_f1,
            'total_parameters': total_params,
            'training_samples': len(train_texts),
            'training_duration_hours': total_duration/3600,
            'dataset_size': '10k_samples',
            'training_type': 'overnight_large_dataset'
        }
        torch.save(checkpoint, best_model_path)
        
        # Save vocabulary and model info
        vocab_path = models_dir / "vocab.pkl"
        data_processor.save_vocabulary(str(vocab_path))
        
        model_info_path = models_dir / "model_info.json"
        save_model_info(model, config, str(model_info_path))
        
        # Copy to main directory
        main_models_dir = Path("models")
        main_models_dir.mkdir(exist_ok=True)
        
        shutil.copy2(best_model_path, main_models_dir / "best_model.pt")
        shutil.copy2(vocab_path, main_models_dir / "vocab.pkl")
        shutil.copy2(model_info_path, main_models_dir / "model_info.json")
        
        # Final test evaluation
        print(f"\n🧪 Final Test Evaluation...")
        test_metrics = trainer.evaluate(dataloaders['test'])
        
        final_metrics = {
            'best_val_f1': trainer.best_val_f1,
            'test_f1_macro': test_metrics['f1_macro'],
            'test_accuracy': test_metrics['accuracy'],
            'total_parameters': total_params,
            'training_samples': len(train_texts),
            'training_duration_hours': total_duration/3600,
            'dataset_scale': '10k_samples'
        }
        
        # Finish experiment
        consolidated_tracker.save_model(experiment_id, model, "overnight_large_model", additional_info=final_metrics)
        consolidated_tracker.finish_experiment(experiment_id, final_metrics=final_metrics)
        
        # Final summary
        print(f"\n🎉 OVERNIGHT SUCCESS!")
        print(f"=" * 60)
        print(f"🆔 Experiment: {experiment_id}")
        print(f"🏆 Best Val F1: {trainer.best_val_f1:.4f}")
        print(f"🧪 Test F1: {test_metrics['f1_macro']:.4f}")
        print(f"📊 Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"⏱️ Training Time: {total_duration/3600:.1f} hours")
        print(f"📈 Dataset: {len(train_texts):,} training samples")
        print(f"🏗️ Model: {total_params:,} parameters")
        print(f"💾 All models saved and ready for use")
        print(f"=" * 60)
        
        return True, experiment_id, final_metrics
        
    except Exception as e:
        print(f"❌ Overnight training failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

if __name__ == "__main__":
    print("🌙 OVERNIGHT LARGE DATASET TRAINING")
    print("Perfect for running while you sleep!")
    print(f"Starting at: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    success, exp_id, metrics = overnight_large_training()
    
    if success:
        print(f"\n🌅 GOOD MORNING! Training completed successfully!")
        print(f"🏆 Your model achieved {metrics['test_f1_macro']:.4f} F1 on {metrics['training_samples']:,} samples")
        print(f"⏱️ Training took {metrics['training_duration_hours']:.1f} hours")
        print(f"\n☕ Ready to test with: python predict.py --interactive")
    else:
        print(f"\n😴 Training encountered issues - check logs above")
