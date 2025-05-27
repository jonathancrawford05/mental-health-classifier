#!/usr/bin/env python3
"""
Main training script for mental health classifier.

This script orchestrates the entire training pipeline:
1. Load and validate configuration
2. Setup data processing and model
3. Train the model
4. Evaluate performance
5. Save results
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from sklearn.model_selection import train_test_split

from src.models import MentalHealthClassifier, create_model
from src.data import DataProcessor, create_sample_data
from src.training import create_trainer
from src.utils import (
    load_config, setup_logging, set_random_seeds, get_device,
    validate_config, print_model_summary, print_data_summary
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Mental Health Classifier")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "simple-cpu"],
        default="standard",
        help="Training mode: standard (full pipeline) or simple-cpu (proven CPU approach)"
    )
    
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample dataset for testing"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Size of sample dataset to create"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--experiment-summary",
        action="store_true",
        help="Show experiment summary before training"
    )
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup training device."""
    if device_arg == "auto":
        device = get_device()
    else:
        device = torch.device(device_arg)
    
    logging.info(f"Using device: {device}")
    return device


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    validate_config(config)
    
    # Setup logging
    setup_logging(
        log_level=config.get('logging', {}).get('log_level', 'INFO'),
        log_file=config.get('logging', {}).get('log_file')
    )
    
    logging.info("Starting Mental Health Classifier Training")
    logging.info(f"Configuration loaded from: {args.config}")
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # For standard mode, just print a recommendation and continue with original flow
    print("⚠️  Standard mode is deprecated. Use --mode simple-cpu for better results.")
    print("   Continuing with original training approach...\n")
    
    # Setup device
    device = setup_device(args.device)
    
    # Initialize data processor
    data_processor = DataProcessor(config['data'])
    
    # Load data
    logging.info("Loading training data...")
    train_texts, train_labels = data_processor.load_data(config['data']['train_path'])
    
    logging.info("Loading validation data...")
    val_texts, val_labels = data_processor.load_data(config['data']['val_path'])
    
    test_texts, test_labels = None, None
    if os.path.exists(config['data']['test_path']):
        logging.info("Loading test data...")
        test_texts, test_labels = data_processor.load_data(config['data']['test_path'])
    
    # Print data summary
    print_data_summary(
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels,
        config['labels']['label_names']
    )
    
    # Build vocabulary
    logging.info("Building vocabulary...")
    data_processor.build_vocabulary(train_texts)
    
    # Update vocab size in config
    config['model']['vocab_size'] = len(data_processor.vocab)
    logging.info(f"Vocabulary size: {config['model']['vocab_size']}")
    
    # Create data loaders
    logging.info("Creating data loaders...")
    dataloaders = data_processor.create_dataloaders(
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels
    )
    
    # Create model
    logging.info("Initializing model...")
    model = create_model(config['model'])
    
    # Print model summary
    print_model_summary(model, config['model'])
    
    # Calculate class weights for imbalanced data
    class_weights = data_processor.get_class_weights(train_labels)
    logging.info(f"Class weights: {class_weights}")
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        config=config['training'],
        device=device,
        class_weights=class_weights
    )
    
    # Train model
    logging.info("Starting training...")
    trainer.train(dataloaders['train'], dataloaders['val'])
    
    # Save vocabulary and model info
    os.makedirs(config['paths']['model_save_dir'], exist_ok=True)
    data_processor.save_vocabulary(
        os.path.join(config['paths']['model_save_dir'], 'vocab.pkl')
    )
    
    from src.utils import save_model_info
    save_model_info(
        model, config,
        os.path.join(config['paths']['model_save_dir'], 'model_info.json')
    )
    
    # Evaluate on test set if available
    if test_texts is not None:
        logging.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(
            dataloaders['test'],
            save_plots=True,
            save_dir='results/'
        )
        
        logging.info("Test Results:")
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                logging.info(f"{metric}: {value:.4f}")
    
    # Save training history
    trainer.save_training_history(
        os.path.join(config['paths']['model_save_dir'], 'training_history.json')
    )
    
    logging.info("Training completed successfully!")
    return True


if __name__ == "__main__":
    main()
