#!/usr/bin/env python3
"""
Simple test script to isolate the training issue.
"""
# %%
import sys
from pathlib import Path
import torch
# %%
# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from models import create_model
from data import DataProcessor, create_sample_data
from training import create_trainer
from utils import set_random_seeds

def test_training():
    """Test training with minimal setup."""
    print("=== TESTING TRAINING LOOP ===")
    
    # Set random seeds
    set_random_seeds(42)
    
    # Create tiny dataset
    print("Creating tiny dataset...")
    temp_file = "tiny_test.csv"
    create_sample_data(temp_file, num_samples=60)  # Just 60 samples
    
    try:
        # Simple configs
        data_config = {
            'text_column': 'text',
            'label_column': 'label',
            'max_length': 64,
            'batch_size': 8
        }
        
        model_config = {
            'vocab_size': 100,  # Will be updated
            'n_embd': 32,
            'num_heads': 2,
            'n_layer': 1,  # Very small model
            'num_classes': 3,
            'max_seq_length': 64,
            'dropout': 0.1
        }
        
        training_config = {
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'num_epochs': 1,  # Just 1 epoch
            'batch_size': 8
        }
        
        # Process data
        print("Processing data...")
        processor = DataProcessor(data_config)
        texts, labels = processor.load_data(temp_file)
        processor.build_vocabulary(texts)
        
        # Update vocab size
        model_config['vocab_size'] = len(processor.vocab)
        print(f"Vocabulary size: {len(processor.vocab)}")
        
        # Split data manually
        split_idx = len(texts) * 2 // 3
        train_texts = texts[:split_idx]
        train_labels = labels[:split_idx]
        val_texts = texts[split_idx:]
        val_labels = labels[split_idx:]
        
        print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")
        
        # Create dataloaders
        dataloaders = processor.create_dataloaders(
            train_texts, train_labels,
            val_texts, val_labels
        )
        
        print(f"Train batches: {len(dataloaders['train'])}")
        print(f"Val batches: {len(dataloaders['val'])}")
        
        # Test one batch
        print("Testing data loading...")
        train_iter = iter(dataloaders['train'])
        batch = next(train_iter)
        print(f"Batch shapes: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}")
        
        # Create model
        print("Creating model...")
        device = torch.device('cpu')  # Force CPU
        model = create_model(model_config)
        
        # Test model forward pass
        print("Testing model forward pass...")
        model.eval()
        with torch.no_grad():
            outputs = model(batch['input_ids'], batch['attention_mask'])
            print(f"Model output shape: {outputs['logits'].shape}")
        
        # Create trainer
        print("Creating trainer...")
        trainer = create_trainer(model, training_config, device)
        
        # Try training
        print("Starting training...")
        trainer.train(dataloaders['train'], dataloaders['val'])
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    test_training()
