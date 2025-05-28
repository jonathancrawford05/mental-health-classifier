#!/usr/bin/env python3
"""
Minimal CPU Training - No Experiment Tracking
Just train and save the model properly
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
import shutil

def minimal_cpu_training():
    """Minimal CPU training without experiment tracking."""
    
    print("🖥️  Minimal CPU Training - No Tracking")
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
        
        # Use proven configuration
        config = {
            "model": {
                "name": "MentalHealthClassformer_Minimal",
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
        
        print(f"✅ Configuration set")
        
        # Data processing
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
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Update config for trainer to save in models directory
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
        
        # Manual model saving to ensure it works
        print(f"💾 Saving model components...")
        
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
        
        # Get final metrics
        final_metrics = {
            'accuracy': trainer.history['val_acc'][-1] if trainer.history['val_acc'] else 0,
            'f1_macro': trainer.history['val_f1'][-1] if trainer.history['val_f1'] else 0,
            'best_val_f1': trainer.best_val_f1,
            'total_parameters': total_params,
            'vocab_size': len(data_processor.vocab)
        }
        
        print(f"\n🎉 SUCCESS! Results:")
        print(f"   Model Parameters: {total_params:,}")
        print(f"   Vocabulary Size: {len(data_processor.vocab)}")
        print(f"   Accuracy: {final_metrics['accuracy']:.3f}")
        print(f"   F1-Macro: {final_metrics['f1_macro']:.3f}")
        print(f"   Best Val F1: {final_metrics['best_val_f1']:.3f}")
        
        print(f"\n📁 Model files saved:")
        print(f"   • {best_model_path}")
        print(f"   • {vocab_path}")
        print(f"   • {model_info_path}")
        
        return True, final_metrics
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_inference():
    """Test that the saved model works for inference."""
    
    print(f"\n🔮 Testing Inference...")
    
    try:
        from predict import MentalHealthPredictor
        
        predictor = MentalHealthPredictor()
        predictor.load_model()
        
        test_texts = [
            "I feel hopeless and can't see any way forward",
            "I'm constantly worried about everything",
            "I've been thinking about ending my life"
        ]
        
        for text in test_texts:
            prediction, probs = predictor.predict(text)
            confidence = max(probs.values())
            
            print(f"\n   Text: {text[:50]}...")
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {confidence:.3f}")
        
        print(f"\n✅ Inference test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Inference test FAILED: {e}")
        return False

if __name__ == "__main__":
    print("🚀 MINIMAL CPU TRAINING TEST")
    print("=" * 50)
    
    # Test training
    training_success, metrics = minimal_cpu_training()
    
    if training_success:
        print(f"\n✅ Training completed successfully!")
        
        # Test inference
        inference_success = test_inference()
        
        if inference_success:
            print(f"\n🎉 COMPLETE SUCCESS!")
            print(f"   • Training works")
            print(f"   • Model saving works")
            print(f"   • Inference works")
            print(f"\n🎯 CPU training is ready for production!")
        else:
            print(f"\n🟡 Training works but inference needs attention")
    else:
        print(f"\n❌ Training failed - check errors above")
