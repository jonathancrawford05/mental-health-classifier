"""
Fixed tests for mental health classifier components.
"""

import sys
import os
from pathlib import Path
import torch

# The imports should work now since we're adding src to path in run_tests.py
from models import MentalHealthClassifier, create_model
from data import DataProcessor, ClinicalTextPreprocessor, create_sample_data
from training import create_trainer
from utils import load_config, set_random_seeds, get_device


def test_clinical_text_preprocessor():
    """Test clinical text preprocessing."""
    print("Testing clinical text preprocessor...")
    
    preprocessor = ClinicalTextPreprocessor()
    
    # Test basic preprocessing
    text = "Pt c/o depression w/ SI"
    processed = preprocessor.preprocess(text)
    
    print(f"Original: {text}")
    print(f"Processed: {processed}")
    
    # Check individual expansions
    assert "patient" in processed, f"'pt' not expanded to 'patient' in: {processed}"
    assert "complains of" in processed, f"'c/o' not expanded to 'complains of' in: {processed}"
    assert "suicidal ideation" in processed, f"'si' not expanded to 'suicidal ideation' in: {processed}"
    
    # Test the w/ expansion more specifically
    test_text_2 = "Patient w/ depression"
    processed_2 = preprocessor.preprocess(test_text_2)
    print(f"Test 2 - Original: {test_text_2}")
    print(f"Test 2 - Processed: {processed_2}")
    
    # The w/ should be expanded to with
    assert "with" in processed_2, f"'w/' not expanded to 'with' in: {processed_2}"
    
    print("✓ Clinical text preprocessor test passed")


def test_model_creation():
    """Test model creation and forward pass."""
    print("Testing model creation...")
    
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
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    output = model(input_ids, attention_mask)
    
    assert 'logits' in output
    assert output['logits'].shape == (batch_size, config['num_classes'])
    
    print(f"✓ Model forward pass successful, output shape: {output['logits'].shape}")


def test_data_processor():
    """Test data processing pipeline."""
    print("Testing data processor...")
    
    # Create temporary sample data
    temp_file = "test_sample.csv"
    requested_samples = 100
    create_sample_data(temp_file, num_samples=requested_samples)
    
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
        
        print(f"Requested {requested_samples} samples, got {len(texts)} texts and {len(labels)} labels")
        
        # Check that we got the expected number (should be exactly requested_samples now)
        assert len(texts) == requested_samples, f"Expected {requested_samples} texts, got {len(texts)}"
        assert len(labels) == requested_samples, f"Expected {requested_samples} labels, got {len(labels)}"
        assert all(isinstance(label, int) for label in labels)
        
        print(f"✓ Loaded {len(texts)} texts and {len(labels)} labels")
        
        # Build vocabulary
        processor.build_vocabulary(texts)
        
        assert processor.vocab is not None
        assert len(processor.vocab) > 0
        
        print(f"✓ Built vocabulary with {len(processor.vocab)} tokens")
        
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
        
        print(f"✓ Created dataset and dataloader successfully")
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_integration():
    """Test full pipeline integration."""
    print("Testing full integration...")
    
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
        
        print(f"✓ Data processed, vocabulary size: {len(processor.vocab)}")
        
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
        
        print(f"✓ Created dataloaders with {len(train_texts)} train, {len(val_texts)} val samples")
        
        # Create trainer
        device = torch.device('cpu')  # Use CPU for testing
        trainer = create_trainer(model, config['training'], device)
        
        print("✓ Created trainer, starting mini training...")
        
        # Train for one epoch (this might take a moment)
        trainer.train(dataloaders['train'], dataloaders['val'])
        
        # Test prediction
        sample_text = "I feel very sad and hopeless"
        prediction = trainer.predict_text(sample_text, processor)
        
        assert prediction in processor.label_names
        
        print(f"✓ Training completed, prediction for test text: '{prediction}'")
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("RUNNING MENTAL HEALTH CLASSIFIER TESTS")
    print("=" * 60)
    
    try:
        test_clinical_text_preprocessor()
        print()
        
        test_model_creation()
        print()
        
        test_data_processor()
        print()
        
        test_integration()
        print()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
