"""
Basic tests for mental health classifier components.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

import torch
import pytest

from models import MentalHealthClassifier, create_model
from data import DataProcessor, ClinicalTextPreprocessor, create_sample_data
from training import create_trainer
from utils import load_config, set_random_seeds, get_device


def test_clinical_text_preprocessor():
    """Test clinical text preprocessing."""
    preprocessor = ClinicalTextPreprocessor()
    
    # Test basic preprocessing
    text = "Pt c/o depression w/ SI"
    processed = preprocessor.preprocess(text)
    
    assert "patient" in processed
    assert "complains of" in processed
    assert "with" in processed
    assert "suicidal ideation" in processed


def test_model_creation():
    """Test model creation and forward pass."""
    config = {
        'vocab_size': 1000,
        'n_embd': 64,
        'num_heads': 4,
        'n_layer': 2,
        'num_classes': 3,
        'max_seq_length': 128,
        'dropout': 0.1
    }
    
    model = create_model(config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    output = model(input_ids, attention_mask)
    
    assert 'logits' in output
    assert output['logits'].shape == (batch_size, config['num_classes'])


def test_data_processor():
    """Test data processing pipeline."""
    # Create temporary sample data
    temp_file = "test_sample.csv"
    create_sample_data(temp_file, num_samples=100)
    
    try:
        config = {
            'text_column': 'text',
            'label_column': 'label',
            'max_length': 128,
            'batch_size': 16
        }
        
        processor = DataProcessor(config)
        
        # Load data
        texts, labels = processor.load_data(temp_file)
        
        assert len(texts) == 100
        assert len(labels) == 100
        assert all(isinstance(label, int) for label in labels)
        
        # Build vocabulary
        processor.build_vocabulary(texts)
        
        assert processor.vocab is not None
        assert len(processor.vocab) > 0
        
        # Create dataset
        dataset = processor.create_dataset(texts[:10], labels[:10])
        
        assert len(dataset) == 10
        
        # Test data loader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=4)
        
        batch = next(iter(dataloader))
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' in batch
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_integration():
    """Test full pipeline integration."""
    set_random_seeds(42)
    
    # Create sample data
    temp_file = "integration_test.csv"
    create_sample_data(temp_file, num_samples=50)
    
    try:
        # Configuration
        config = {
            'model': {
                'vocab_size': 500,
                'n_embd': 32,
                'num_heads': 2,
                'n_layer': 2,
                'num_classes': 3,
                'max_seq_length': 64,
                'dropout': 0.1
            },
            'data': {
                'text_column': 'text',
                'label_column': 'label',
                'max_length': 64,
                'batch_size': 8
            },
            'training': {
                'learning_rate': 1e-3,
                'num_epochs': 1,
                'batch_size': 8
            }
        }
        
        # Data processing
        processor = DataProcessor(config['data'])
        texts, labels = processor.load_data(temp_file)
        processor.build_vocabulary(texts)
        
        # Update vocab size
        config['model']['vocab_size'] = len(processor.vocab)
        
        # Create model
        model = create_model(config['model'])
        
        # Create data loader
        train_texts = texts[:30]
        train_labels = labels[:30]
        val_texts = texts[30:]
        val_labels = labels[30:]
        
        dataloaders = processor.create_dataloaders(
            train_texts, train_labels,
            val_texts, val_labels
        )
        
        # Create trainer
        device = torch.device('cpu')  # Use CPU for testing
        trainer = create_trainer(model, config['training'], device)
        
        # Train for one epoch
        trainer.train(dataloaders['train'], dataloaders['val'])
        
        # Test prediction
        sample_text = "I feel very sad and hopeless"
        prediction = trainer.predict_text(sample_text, processor)
        
        assert prediction in processor.label_names
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == "__main__":
    print("Running basic tests...")
    
    try:
        test_clinical_text_preprocessor()
        print("✓ Clinical text preprocessor test passed")
        
        test_model_creation()
        print("✓ Model creation test passed")
        
        test_data_processor()
        print("✓ Data processor test passed")
        
        test_integration()
        print("✓ Integration test passed")
        
        print("\nAll tests passed! 🎉")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise
