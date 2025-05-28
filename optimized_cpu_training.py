#!/usr/bin/env python3
"""
Optimized CPU Training with Intelligent Early Stopping
Building on large experiment success (19M params, 1.0 F1 in 9.4 hours)
Target: Same performance in 2-3 hours with smart stopping criteria
"""

import os
import sys
from pathlib import Path
import time
from datetime import datetime, timedelta

# Environment setup
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import json
import shutil
import numpy as np
from typing import Dict, List, Optional

class EarlyStopping:
    """
    Intelligent early stopping for mental health model training.
    Combines multiple stopping criteria for optimal efficiency.
    """
    
    def __init__(self, 
                 patience: int = 3,
                 min_delta: float = 0.001,
                 performance_threshold: float = 0.95,
                 max_training_hours: float = 4.0,
                 restore_best_weights: bool = True):
        """
        Initialize early stopping with multiple criteria.
        
        Args:
            patience: Epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            performance_threshold: Stop when F1 exceeds this
            max_training_hours: Maximum training time
            restore_best_weights: Whether to restore best model
        """
        self.patience = patience
        self.min_delta = min_delta
        self.performance_threshold = performance_threshold
        self.max_training_hours = max_training_hours
        self.restore_best_weights = restore_best_weights
        
        # State tracking
        self.best_score = -np.inf
        self.best_epoch = 0
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.start_time = None
        self.stop_reason = None
        
    def start_training(self):
        """Mark the start of training."""
        self.start_time = time.time()
        print(f"🕐 Early stopping active - Max time: {self.max_training_hours:.1f}h, "
              f"Patience: {self.patience}, Threshold: {self.performance_threshold}")
    
    def __call__(self, 
                 epoch: int,
                 val_f1: float,
                 model_weights: Optional[Dict] = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            epoch: Current epoch number
            val_f1: Validation F1 score
            model_weights: Current model weights
            
        Returns:
            True if training should stop
        """
        # Check time limit
        if self.start_time:
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours >= self.max_training_hours:
                self.stop_reason = f"time_limit_reached_{elapsed_hours:.1f}h"
                self.stopped_epoch = epoch
                print(f"⏰ Time limit reached ({elapsed_hours:.1f}h) - stopping training")
                return True
        
        # Check performance threshold
        if val_f1 >= self.performance_threshold:
            self.stop_reason = f"performance_threshold_reached_{val_f1:.4f}"
            self.stopped_epoch = epoch
            print(f"🎯 Performance threshold reached (F1: {val_f1:.4f}) - stopping training")
            return True
        
        # Check improvement
        if val_f1 > self.best_score + self.min_delta:
            self.best_score = val_f1
            self.best_epoch = epoch
            self.wait = 0
            if model_weights and self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model_weights.items()}
            print(f"📈 New best F1: {val_f1:.4f} (epoch {epoch})")
        else:
            self.wait += 1
            print(f"⏳ No improvement for {self.wait}/{self.patience} epochs (current: {val_f1:.4f}, best: {self.best_score:.4f})")
            
            if self.wait >= self.patience:
                self.stop_reason = f"no_improvement_{self.patience}_epochs"
                self.stopped_epoch = epoch
                print(f"🛑 Early stopping triggered - no improvement for {self.patience} epochs")
                return True
        
        return False
    
    def get_summary(self) -> Dict:
        """Get summary of early stopping results."""
        return {
            'stopped': self.stopped_epoch > 0,
            'stopped_epoch': self.stopped_epoch,
            'best_epoch': self.best_epoch,
            'best_score': self.best_score,
            'stop_reason': self.stop_reason,
            'epochs_saved': max(0, 12 - self.stopped_epoch) if self.stopped_epoch > 0 else 0
        }

def optimized_cpu_training():
    """Optimized CPU training with intelligent early stopping."""
    
    print("🚀 OPTIMIZED CPU TRAINING WITH EARLY STOPPING")
    print("=" * 70)
    print("Building on large experiment: 19M params, 1.0 F1 in 9.4 hours")
    print("Target: Same performance in 2-3 hours with intelligent stopping")
    print("=" * 70)
    
    # Optimal CPU setup
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
        
        # OPTIMIZED CONFIGURATION
        # Same architecture as large experiment but with optimization focus
        config = {
            "model": {
                "name": "MentalHealthTransformer_Optimized",
                "vocab_size": 20000,
                "n_embd": 512,
                "num_heads": 8,
                "n_layer": 6,
                "num_classes": 3,
                "max_seq_length": 512,
                "dropout": 0.15
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 2e-05,
                "weight_decay": 0.03,
                "num_epochs": 15,      # Allow more epochs but expect early stopping
                "warmup_steps": 300,
                "gradient_clip_norm": 1.0,
                "save_every": 100,     # More frequent saves
                "eval_every": 25       # Very frequent evaluation for early stopping
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
                "patience": 3,
                "min_delta": 0.001,
                "performance_threshold": 0.95,
                "max_training_hours": 4.0
            }
        }
        
        # Start experiment
        experiment_id = consolidated_tracker.start_experiment(
            experiment_name="optimized_cpu_training",
            description="Optimized CPU training with intelligent early stopping - targeting 2-3h vs 9.4h",
            config=config,
            tags=["cpu", "optimized", "early-stopping", "production"]
        )
        
        print(f"🧪 Started experiment: {experiment_id}")
        
        # Setup directories
        exp_dir = Path("experiments") / experiment_id
        models_dir = exp_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Use existing large dataset (5K samples)
        print(f"\n📊 Dataset Loading:")
        data_processor = DataProcessor(config['data'])
        
        # Check if large dataset exists, create if needed
        import pandas as pd
        if not Path("data/train.csv").exists() or len(pd.read_csv("data/train.csv")) < 3000:
            print(f"   🔄 Creating optimized dataset...")
            from src.data import create_sample_data
            from sklearn.model_selection import train_test_split
            
            create_sample_data("data/optimized_train.csv", 5000)
            
            df = pd.read_csv("data/optimized_train.csv")
            train_val_df, test_df = train_test_split(
                df, test_size=0.15, random_state=42, stratify=df['label']
            )
            train_df, val_df = train_test_split(
                train_val_df, test_size=0.18, random_state=42, stratify=train_val_df['label']
            )
            
            train_df.to_csv("data/train.csv", index=False)
            val_df.to_csv("data/val.csv", index=False)
            test_df.to_csv("data/test.csv", index=False)
        
        # Load data
        train_texts, train_labels = data_processor.load_data('data/train.csv')
        val_texts, val_labels = data_processor.load_data('data/val.csv')
        test_texts, test_labels = data_processor.load_data('data/test.csv')
        
        print(f"   ✅ Dataset loaded:")
        print(f"      Training: {len(train_texts)} samples")
        print(f"      Validation: {len(val_texts)} samples")
        print(f"      Test: {len(test_texts)} samples")
        
        # Build vocabulary
        data_processor.build_vocabulary(train_texts)
        actual_vocab_size = len(data_processor.vocab)
        config['model']['vocab_size'] = actual_vocab_size
        
        print(f"   ✅ Vocabulary: {actual_vocab_size} tokens")
        
        # Create data loaders
        dataloaders = data_processor.create_dataloaders(
            train_texts, train_labels,
            val_texts, val_labels,
            test_texts, test_labels
        )
        
        # Create model
        print(f"\n🏗️ Model Architecture:")
        model = create_model(config['model']).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / (1024**2)
        
        print(f"   Parameters: {total_params:,} ({model_size_mb:.1f} MB)")
        print(f"   Architecture: {config['model']['n_layer']}L-{config['model']['n_embd']}D-{config['model']['num_heads']}H")
        
        # Setup early stopping
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
        
        # OPTIMIZED TRAINING LOOP with Early Stopping
        print(f"\n🏋️‍♂️ Optimized Training with Early Stopping:")
        print(f"   Max epochs: {config['training']['num_epochs']}")
        print(f"   Early stopping patience: {config['early_stopping']['patience']}")
        print(f"   Performance threshold: {config['early_stopping']['performance_threshold']}")
        print(f"   Max training time: {config['early_stopping']['max_training_hours']} hours")
        
        early_stopping.start_training()
        start_time = datetime.now()
        
        # Modified training loop with early stopping integration
        num_epochs = config['training']['num_epochs']
        best_model_state = None
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training phase
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
                
                # Progress update
                if (batch_idx + 1) % 50 == 0:
                    elapsed = time.time() - epoch_start
                    print(f"    Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloaders['train'])}: "
                          f"Loss={loss.item():.4f}, Time={elapsed:.1f}s")
            
            avg_train_loss = train_loss / train_batches
            
            # Validation phase
            val_metrics = trainer.evaluate(dataloaders['val'])
            val_f1 = val_metrics['f1_macro']
            val_acc = val_metrics['accuracy']
            
            # Update trainer history
            trainer.history['train_loss'].append(avg_train_loss)
            trainer.history['val_loss'].append(val_metrics['loss'])
            trainer.history['val_acc'].append(val_acc)
            trainer.history['val_f1'].append(val_f1)
            
            # Update best model if needed
            if val_f1 > trainer.best_val_f1:
                trainer.best_val_f1 = val_f1
                trainer.best_model_state = model.state_dict().copy()
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'best_val_f1': trainer.best_val_f1,
                    'config': config
                }
                torch.save(checkpoint, models_dir / 'best_model.pt')
            
            epoch_time = time.time() - epoch_start
            elapsed_total = (datetime.now() - start_time).total_seconds() / 3600
            
            print(f"\\n📊 Epoch {epoch+1}/{num_epochs} Results:")
            print(f"   Train Loss: {avg_train_loss:.4f}")
            print(f"   Val Loss: {val_metrics['loss']:.4f}")
            print(f"   Val Accuracy: {val_acc:.4f}")
            print(f"   Val F1: {val_f1:.4f}")
            print(f"   Best F1: {trainer.best_val_f1:.4f}")
            print(f"   Epoch Time: {epoch_time/60:.1f} min")
            print(f"   Total Time: {elapsed_total:.1f}h")
            
            # Log metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_acc,
                'val_f1_macro': val_f1,
                'best_val_f1': trainer.best_val_f1,
                'epoch_time_minutes': epoch_time / 60,
                'total_time_hours': elapsed_total
            }
            consolidated_tracker.log_metrics(experiment_id, epoch_metrics, epoch=epoch+1)
            
            # Check early stopping
            should_stop = early_stopping(epoch + 1, val_f1, model.state_dict())
            if should_stop:
                print(f"\\n🛑 Early stopping triggered!")
                break
        
        # Training completed
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Early stopping summary
        es_summary = early_stopping.get_summary()
        
        print(f"\\n✅ Training Completed!")
        print(f"   Total time: {total_duration/60:.1f} minutes ({total_duration/3600:.1f} hours)")
        print(f"   Epochs completed: {epoch + 1}/{num_epochs}")
        if es_summary['stopped']:
            print(f"   Early stopping: {es_summary['stop_reason']}")
            print(f"   Epochs saved: {es_summary['epochs_saved']}")
            print(f"   Best epoch: {es_summary['best_epoch']}")
        
        # Restore best weights if early stopping was used
        if early_stopping.best_weights and early_stopping.restore_best_weights:
            model.load_state_dict(early_stopping.best_weights)
            print(f"   ✅ Restored best model weights from epoch {early_stopping.best_epoch}")
        
        # Save all components
        print(f"\\n💾 Saving Optimized Model:")
        
        # Enhanced checkpoint
        best_model_path = models_dir / "best_model.pt"
        checkpoint = {
            'epoch': early_stopping.best_epoch if es_summary['stopped'] else epoch + 1,
            'model_state_dict': trainer.best_model_state or model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'best_val_f1': trainer.best_val_f1,
            'config': config,
            'total_parameters': total_params,
            'training_duration_minutes': total_duration/60,
            'early_stopping_summary': es_summary,
            'optimization_type': 'early_stopping_cpu_training'
        }
        torch.save(checkpoint, best_model_path)
        
        # Save other components
        vocab_path = models_dir / "vocab.pkl"
        data_processor.save_vocabulary(str(vocab_path))
        
        model_info_path = models_dir / "model_info.json"
        save_model_info(model, config, str(model_info_path))
        
        history_path = models_dir / "training_history.json"
        trainer.save_training_history(str(history_path))
        
        # Copy to main directory
        main_models_dir = Path("models")
        main_models_dir.mkdir(exist_ok=True)
        
        shutil.copy2(best_model_path, main_models_dir / "best_model.pt")
        shutil.copy2(vocab_path, main_models_dir / "vocab.pkl")
        shutil.copy2(model_info_path, main_models_dir / "model_info.json")
        
        # Final evaluation
        print(f"\\n📊 Final Evaluation:")
        
        final_metrics = {
            'best_val_f1': trainer.best_val_f1,
            'final_val_accuracy': trainer.history['val_acc'][-1] if trainer.history['val_acc'] else 0,
            'total_parameters': total_params,
            'model_size_mb': model_size_mb,
            'training_duration_minutes': total_duration/60,
            'training_duration_hours': total_duration/3600,
            'epochs_completed': epoch + 1,
            'early_stopping_triggered': es_summary['stopped'],
            'early_stopping_reason': es_summary['stop_reason'],
            'epochs_saved': es_summary.get('epochs_saved', 0),
            'time_efficiency_improvement': f"{9.4 / (total_duration/3600):.1f}x_faster" if total_duration > 0 else "N/A"
        }
        
        # Test evaluation
        if test_texts:
            print(f"   🧪 Test set evaluation...")
            test_metrics = trainer.evaluate(dataloaders['test'])
            final_metrics.update({
                'test_accuracy': test_metrics['accuracy'],
                'test_f1_macro': test_metrics['f1_macro'],
                'test_precision_macro': test_metrics['precision_macro'],
                'test_recall_macro': test_metrics['recall_macro']
            })
        
        # Finish experiment
        consolidated_tracker.save_model(experiment_id, model, "optimized_model", additional_info=final_metrics)
        consolidated_tracker.finish_experiment(experiment_id, final_metrics=final_metrics)
        
        # Results summary
        print(f"\\n🎉 OPTIMIZATION RESULTS:")
        print(f"=" * 70)
        print(f"🆔 Experiment: {experiment_id}")
        print(f"⏱️ Training Time: {total_duration/60:.1f} min ({total_duration/3600:.1f}h)")
        print(f"🏆 Best F1: {trainer.best_val_f1:.4f}")
        print(f"📊 Final Accuracy: {final_metrics['final_val_accuracy']:.4f}")
        if 'test_f1_macro' in final_metrics:
            print(f"🧪 Test F1: {final_metrics['test_f1_macro']:.4f}")
        print(f"📈 Epochs: {epoch + 1}/{num_epochs}")
        if es_summary['stopped']:
            print(f"🛑 Early Stop: {es_summary['stop_reason']}")
            print(f"💡 Epochs Saved: {es_summary['epochs_saved']}")
            time_saved_hours = es_summary['epochs_saved'] * (total_duration / 3600 / (epoch + 1))
            print(f"⏰ Est. Time Saved: {time_saved_hours:.1f}h")
        
        # Efficiency comparison
        original_time = 9.4  # hours from large experiment
        current_time = total_duration / 3600
        if current_time > 0:
            efficiency_gain = original_time / current_time
            print(f"🚀 Efficiency Gain: {efficiency_gain:.1f}x faster than large experiment")
            print(f"   Large exp: 9.4h → Optimized: {current_time:.1f}h")
        
        print(f"=" * 70)
        
        return True, experiment_id, final_metrics
        
    except Exception as e:
        print(f"❌ Optimized training failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

if __name__ == "__main__":
    print("🎯 OPTIMIZED CPU TRAINING WITH EARLY STOPPING")
    print("Target: Same 1.0 F1 performance in 2-3 hours vs 9.4 hours")
    print()
    
    success, exp_id, metrics = optimized_cpu_training()
    
    if success:
        print(f"\\n✅ OPTIMIZATION SUCCESS!")
        print(f"🆔 {exp_id}")
        print(f"🏆 F1: {metrics['best_val_f1']:.4f}")
        print(f"⏱️ Time: {metrics['training_duration_hours']:.1f}h")
        if metrics.get('early_stopping_triggered'):
            print(f"🛑 Early stopped: {metrics['early_stopping_reason']}")
            print(f"💡 Saved {metrics['epochs_saved']} epochs")
        
        # Quick test
        print(f"\\n🔮 Testing optimized model...")
        try:
            from predict import MentalHealthPredictor
            predictor = MentalHealthPredictor()
            predictor.load_model()
            
            test_text = "I feel overwhelmed and hopeless about everything"
            prediction, probs = predictor.predict(test_text)
            
            print(f"   Test: '{test_text}'")
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {max(probs.values()):.3f}")
            
            print(f"\\n🎉 OPTIMIZED MODEL READY!")
            
        except Exception as e:
            print(f"⚠️ Model test issue: {e}")
            
    else:
        print(f"\\n❌ Optimization failed")
