#!/usr/bin/env python3
"""
Optimized CPU Training Script

Uses smaller, more efficient model architecture better suited
for the current dataset size and CPU training.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Environment setup
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import json
import shutil
from optimized_config import create_optimized_config

def optimized_cpu_training():
    """Train optimized model on CPU with efficient architecture."""
    
    print("🚀 OPTIMIZED CPU TRAINING")
    print("=" * 50)
    print("Smaller, more efficient model for better CPU performance")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Setup CPU training
    torch.set_num_threads(6)
    device = torch.device('cpu')
    print(f"✅ Device: {device}")
    print(f"✅ Threads: {torch.get_num_threads()}")
    
    try:
        # Import modules
        from src.data import DataProcessor, create_sample_data
        from src.models import create_model
        from src.training import create_trainer
        from src.utils import save_model_info
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        # Get optimized configuration
        config = create_optimized_config()
        
        print(f"\\n📋 OPTIMIZED MODEL CONFIGURATION:")
        print(f"   • Parameters: ~{config['model']['estimated_parameters']:,}")
        print(f"   • Size: ~{config['model']['estimated_size_mb']:.1f} MB")
        print(f"   • Architecture: {config['model']['n_layer']}L-{config['model']['n_embd']}D-{config['model']['num_heads']}H")
        print(f"   • Max length: {config['model']['max_seq_length']}")
        print(f"   • Batch size: {config['training']['batch_size']}")
        
        # Setup directories
        models_dir = Path(config['paths']['model_save_dir'])
        results_dir = Path(config['paths']['results_dir'])
        models_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create moderate dataset (5K samples - good balance)
        print(f"\\n📊 Creating Optimized Dataset:")
        data_processor = DataProcessor(config['data'])
        
        print(f"   🔄 Generating 5,000 samples for optimized training...")
        create_sample_data("data/optimized_dataset.csv", 5000)
        
        # Load and split dataset
        df = pd.read_csv("data/optimized_dataset.csv")
        
        # 70/20/10 split
        train_val_df, test_df = train_test_split(
            df, test_size=0.10, random_state=42, stratify=df['label']
        )
        
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.22, random_state=42, stratify=train_val_df['label']  # ~20% of total
        )
        
        # Save dataset
        train_df.to_csv("data/train.csv", index=False)
        val_df.to_csv("data/val.csv", index=False)
        test_df.to_csv("data/test.csv", index=False)
        
        # Load data
        train_texts, train_labels = data_processor.load_data('data/train.csv')
        val_texts, val_labels = data_processor.load_data('data/val.csv')
        test_texts, test_labels = data_processor.load_data('data/test.csv')
        
        print(f"   ✅ Optimized dataset created:")
        print(f"      Training: {len(train_texts):,} samples")
        print(f"      Validation: {len(val_texts):,} samples")
        print(f"      Test: {len(test_texts):,} samples")
        print(f"      Total: {len(train_texts) + len(val_texts) + len(test_texts):,} samples")
        
        # Build vocabulary
        data_processor.build_vocabulary(train_texts)
        actual_vocab_size = len(data_processor.vocab)
        config['model']['vocab_size'] = actual_vocab_size
        
        print(f"   ✅ Vocabulary: {actual_vocab_size:,} tokens")
        
        # Create data loaders
        dataloaders = data_processor.create_dataloaders(
            train_texts, train_labels,
            val_texts, val_labels,
            test_texts, test_labels
        )
        
        # Create optimized model
        print(f"\\n🏗️ Creating Optimized Model:")
        model = create_model(config['model']).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / (1024**2)
        
        print(f"   Parameters: {total_params:,} ({model_size_mb:.1f} MB)")
        print(f"   Training samples: {len(train_texts):,}")
        print(f"   Batches per epoch: {len(dataloaders['train'])}")
        
        # Estimate training time
        estimated_batch_time = 0.3  # seconds per batch (optimized model)
        estimated_epoch_time = len(dataloaders['train']) * estimated_batch_time / 60
        estimated_total_time = estimated_epoch_time * 10  # assume ~10 epochs
        
        print(f"   Est. epoch time: {estimated_epoch_time:.1f} minutes")
        print(f"   Est. total time: {estimated_total_time:.1f} minutes")
        
        # Calculate class weights
        class_weights = data_processor.get_class_weights(train_labels)
        class_weights = class_weights.to(device)
        
        print(f"   Class weights: {[f'{w:.2f}' for w in class_weights]}")
        
        # Setup trainer
        trainer_config = config['training'].copy()
        trainer_config['model_save_dir'] = str(models_dir)
        
        trainer = create_trainer(
            model=model,
            config=trainer_config,
            device=device,
            class_weights=class_weights
        )
        
        # Training loop
        print(f"\\n🎯 Starting Optimized Training:")
        print(f"   Target F1: {config['training']['early_stopping']['performance_threshold']}")
        print(f"   Max epochs: {config['training']['num_epochs']}")
        print(f"   Expected completion: {datetime.fromtimestamp(datetime.now().timestamp() + estimated_total_time*60).strftime('%H:%M')}")
        
        start_time = datetime.now()
        best_val_f1 = 0.0
        best_model_state = None
        patience_counter = 0
        max_patience = config['training']['early_stopping']['patience']
        target_f1 = config['training']['early_stopping']['performance_threshold']
        
        for epoch in range(config['training']['num_epochs']):
            epoch_start = time.time()
            
            print(f"\\n--- EPOCH {epoch+1}/{config['training']['num_epochs']} ---")
            
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
                
                # Progress update every 20 batches
                if (batch_idx + 1) % 20 == 0:
                    progress = (batch_idx + 1) / len(dataloaders['train']) * 100
                    print(f"   Batch {batch_idx+1:2d}/{len(dataloaders['train']):2d} ({progress:5.1f}%) | Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / train_batches
            
            # Validation phase
            print(f"   Validating...")
            val_metrics = trainer.evaluate(dataloaders['val'])
            val_f1 = val_metrics['f1_macro']
            val_acc = val_metrics['accuracy']
            
            # Check for improvement
            is_best = False
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()
                is_best = True
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_f1': best_val_f1,
                    'config': config
                }, models_dir / 'best_model.pt')
            else:
                patience_counter += 1
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            elapsed_total = (datetime.now() - start_time).total_seconds() / 60
            
            print(f"   Train Loss: {avg_train_loss:.4f}")
            print(f"   Val F1: {val_f1:.4f} {'🌟 NEW BEST!' if is_best else ''}")
            print(f"   Val Accuracy: {val_acc:.4f}")
            print(f"   Best F1: {best_val_f1:.4f}")
            print(f"   Epoch time: {epoch_time/60:.1f}m | Total: {elapsed_total:.1f}m")
            print(f"   Patience: {patience_counter}/{max_patience}")
            
            # Early stopping checks
            if val_f1 >= target_f1:
                print(f"\\n🎯 TARGET F1 ACHIEVED! ({val_f1:.4f} >= {target_f1})")
                break
            
            if patience_counter >= max_patience:
                print(f"\\n⏱️ EARLY STOPPING - No improvement for {max_patience} epochs")
                break
        
        # Training completed
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds() / 60
        
        print(f"\\n✅ OPTIMIZED TRAINING COMPLETED!")
        print(f"   Finished at: {end_time.strftime('%H:%M:%S')}")
        print(f"   Total time: {total_duration:.1f} minutes")
        print(f"   Epochs: {epoch + 1}/{config['training']['num_epochs']}")
        print(f"   Best F1: {best_val_f1:.4f}")
        
        # Load best model for final evaluation
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Final test evaluation
        print(f"\\n🧪 Final Test Evaluation...")
        test_metrics = trainer.evaluate(dataloaders['test'])
        
        print(f"   Test F1: {test_metrics['f1_macro']:.4f}")
        print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   Test Loss: {test_metrics['loss']:.4f}")
        
        # Save everything
        print(f"\\n💾 Saving Optimized Model...")
        
        # Save to optimized directory
        best_model_path = models_dir / "best_model.pt"
        vocab_path = models_dir / "vocab.pkl"
        model_info_path = models_dir / "model_info.json"
        
        # Save vocabulary
        data_processor.save_vocabulary(str(vocab_path))
        
        # Save model info
        save_model_info(model, config, str(model_info_path))
        
        # Copy to main models directory for immediate use
        main_models_dir = Path("models")
        main_models_dir.mkdir(exist_ok=True)
        
        shutil.copy2(best_model_path, main_models_dir / "best_model.pt")
        shutil.copy2(vocab_path, main_models_dir / "vocab.pkl")
        shutil.copy2(model_info_path, main_models_dir / "model_info.json")
        
        # Final summary
        print(f"\\n🎉 OPTIMIZATION SUCCESS!")
        print(f"=" * 50)
        print(f"🏆 Best Val F1: {best_val_f1:.4f}")
        print(f"🧪 Test F1: {test_metrics['f1_macro']:.4f}")
        print(f"📊 Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"⏱️ Training Time: {total_duration:.1f} minutes")
        print(f"📈 Dataset: {len(train_texts):,} samples")
        print(f"🏗️ Model: {total_params:,} parameters ({model_size_mb:.1f} MB)")
        print(f"⚡ {(total_params/1e6):.1f}M params - {total_duration:.0f}min training")
        print(f"💾 Model ready at: models/best_model.pt")
        print(f"=" * 50)
        
        # Test prediction
        print(f"\\n🔮 Testing Prediction...")
        try:
            from predict import MentalHealthPredictor
            predictor = MentalHealthPredictor()
            predictor.load_model()
            
            test_text = "I feel hopeless and can't see any way forward"
            prediction, probs = predictor.predict(test_text)
            
            print(f"   Test: '{test_text}'")
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {max(probs.values()):.3f}")
            print(f"   ✅ Prediction system working!")
        except Exception as e:
            print(f"   ⚠️ Prediction test failed: {e}")
        
        return {
            'success': True,
            'best_val_f1': best_val_f1,
            'test_f1': test_metrics['f1_macro'],
            'test_accuracy': test_metrics['accuracy'],
            'total_params': total_params,
            'training_time_minutes': total_duration,
            'epochs_completed': epoch + 1
        }
        
    except Exception as e:
        print(f"❌ Optimized training failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    print("🚀 OPTIMIZED CPU TRAINING")
    print("Efficient model architecture for better performance")
    print()
    
    results = optimized_cpu_training()
    
    if results['success']:
        print(f"\\n🌟 TRAINING SUCCESSFUL!")
        print(f"   Your optimized model achieved:")
        print(f"   • Validation F1: {results['best_val_f1']:.4f}")
        print(f"   • Test F1: {results['test_f1']:.4f}")
        print(f"   • Training time: {results['training_time_minutes']:.0f} minutes")
        print(f"   • Model size: {results['total_params']/1e6:.1f}M parameters")
        print(f"\\n🎯 Ready to test: python predict.py --interactive")
    else:
        print(f"\\n❌ Training failed: {results.get('error', 'Unknown error')}")
