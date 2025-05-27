#!/usr/bin/env python3
"""
Enhanced training script with comprehensive experiment tracking.
Supports systematic architecture exploration and model comparison.
"""

import os
import sys
import argparse
import logging
import yaml
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
from src.utils.experiment_tracker import tracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Mental Health Classifier with Experiment Tracking")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--experiment-config",
        type=str,
        help="Specific experiment configuration from experiment_configs.yaml (e.g., 'medium_model')"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Custom experiment name (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--description",
        type=str,
        help="Experiment description"
    )
    
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="Tags for experiment tracking"
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
        "--compare-experiments",
        action="store_true",
        help="Compare all completed experiments after training"
    )
    
    return parser.parse_args()


def load_experiment_config(experiment_name: str, base_config: dict) -> dict:
    """Load specific experiment configuration and merge with base config."""
    
    experiment_configs_path = "config/experiment_configs.yaml"
    
    if not os.path.exists(experiment_configs_path):
        raise FileNotFoundError(f"Experiment configs file not found: {experiment_configs_path}")
    
    with open(experiment_configs_path, 'r') as f:
        experiment_configs = yaml.safe_load(f)
    
    if experiment_name not in experiment_configs:
        available = list(experiment_configs.keys())
        raise ValueError(f"Experiment '{experiment_name}' not found. Available: {available}")
    
    exp_config = experiment_configs[experiment_name]
    
    # Deep merge with base config
    merged_config = deep_merge_configs(base_config, exp_config)
    
    return merged_config


def deep_merge_configs(base_config: dict, experiment_config: dict) -> dict:
    """Deep merge experiment config with base config."""
    import copy
    
    result = copy.deepcopy(base_config)
    
    def merge_dict(target, source):
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                merge_dict(target[key], value)
            else:
                target[key] = value
    
    merge_dict(result, experiment_config)
    return result


def setup_device(device_arg: str) -> torch.device:
    """Setup training device."""
    if device_arg == "auto":
        device = get_device()
    else:
        device = torch.device(device_arg)
    
    logging.info(f"Using device: {device}")
    return device


def main():
    """Main training function with experiment tracking."""
    args = parse_args()
    
    # Load base configuration
    config = load_config(args.config)
    
    # Load experiment-specific configuration if provided
    if args.experiment_config:
        config = load_experiment_config(args.experiment_config, config)
        experiment_name = args.experiment_name or args.experiment_config
        description = args.description or config.get('description', f"Training with {args.experiment_config} configuration")
        tags = args.tags or config.get('tags', [])
    else:
        experiment_name = args.experiment_name or "manual_training"
        description = args.description or "Manual training run"
        tags = args.tags or ["manual"]
    
    validate_config(config)
    
    # Setup logging
    setup_logging(
        log_level=config.get('logging', {}).get('log_level', 'INFO'),
        log_file=config.get('logging', {}).get('log_file')
    )
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    config['device'] = str(device)
    
    # Start experiment tracking
    experiment_id = tracker.start_experiment(
        experiment_name=experiment_name,
        description=description,
        config=config,
        tags=tags
    )
    
    logging.info(f"🧪 Experiment ID: {experiment_id}")
    
    try:
        # Create sample data if requested
        if args.create_sample_data:
            logging.info(f"Creating sample dataset with {args.sample_size} samples")
            
            os.makedirs("data", exist_ok=True)
            create_sample_data("data/sample_train.csv", args.sample_size)
            
            # Create train/val/test splits
            import pandas as pd
            df = pd.read_csv("data/sample_train.csv")
            
            # First split: 80% train+val, 20% test
            train_val_df, test_df = train_test_split(
                df, test_size=0.2, random_state=args.seed, 
                stratify=df['label']
            )
            
            # Second split: 80% train, 20% val (of the remaining 80%)
            train_df, val_df = train_test_split(
                train_val_df, test_size=0.25, random_state=args.seed,
                stratify=train_val_df['label']
            )
            
            # Save splits
            train_df.to_csv("data/train.csv", index=False)
            val_df.to_csv("data/val.csv", index=False)
            test_df.to_csv("data/test.csv", index=False)
            
            logging.info("Sample data created and split into train/val/test sets")
            
            # Update config paths
            config['data']['train_path'] = "data/train.csv"
            config['data']['val_path'] = "data/val.csv"
            config['data']['test_path'] = "data/test.csv"
        
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
        
        # Log data statistics
        data_stats = {
            "train_samples": len(train_texts),
            "val_samples": len(val_texts),
            "test_samples": len(test_texts) if test_texts else 0,
            "total_samples": len(train_texts) + len(val_texts) + (len(test_texts) if test_texts else 0)
        }
        tracker.log_metrics(experiment_id, data_stats, step=0)
        
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
        
        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_stats = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Approximate size in MB
        }
        tracker.log_metrics(experiment_id, model_stats, step=0)
        
        # Calculate class weights for imbalanced data
        class_weights = data_processor.get_class_weights(train_labels)
        logging.info(f"Class weights: {class_weights}")
        
        # Create trainer with experiment tracking
        trainer = create_trainer(
            model=model,
            config=config['training'],
            device=device,
            class_weights=class_weights,
            experiment_tracker=(tracker, experiment_id)  # Pass tracker info
        )
        
        # Train model
        logging.info("Starting training...")
        trainer.train(dataloaders['train'], dataloaders['val'])
        
        # Save model
        model_save_dir = f"experiments/{experiment_id}/models"
        os.makedirs(model_save_dir, exist_ok=True)
        
        tracker.save_model(
            experiment_id, 
            model, 
            "final_model",
            additional_info={
                "vocab_size": len(data_processor.vocab),
                "class_weights": class_weights.tolist(),
                "config": config
            }
        )
        
        # Save vocabulary and model info
        data_processor.save_vocabulary(
            os.path.join(model_save_dir, 'vocab.pkl')
        )
        
        # Evaluate on test set if available
        if test_texts is not None:
            logging.info("Evaluating on test set...")
            test_metrics = trainer.evaluate(
                dataloaders['test'],
                save_plots=True,
                save_dir=f'experiments/{experiment_id}/results/'
            )
            
            logging.info("Test Results:")
            for metric, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    logging.info(f"{metric}: {value:.4f}")
            
            # Log final test metrics
            tracker.log_metrics(experiment_id, test_metrics, step=-1)
        
        # Get final training metrics
        final_metrics = getattr(trainer, 'best_metrics', {})
        
        # Finish experiment
        tracker.finish_experiment(
            experiment_id,
            final_metrics=final_metrics,
            notes=f"Completed successfully with {config['model']['n_layer']} layers, {config['model']['n_embd']} embedding dim"
        )
        
        # Interactive prediction example
        if test_texts is not None:
            print("\n" + "="*60)
            print("INTERACTIVE PREDICTION EXAMPLE")
            print("="*60)
            
            # Test with a few examples
            sample_texts = [
                "I feel hopeless and can't see any way out of this darkness",
                "I'm constantly worried about everything and can't relax",
                "I've been thinking about ending my life lately"
            ]
            
            for text in sample_texts:
                prediction, probabilities = trainer.predict_text(
                    text, data_processor, return_probabilities=True
                )
                
                print(f"\nText: {text}")
                print(f"Predicted class: {prediction}")
                print("Probabilities:")
                for label, prob in probabilities.items():
                    print(f"  {label}: {prob:.3f}")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Experiment ID: {experiment_id}")
        print(f"Results saved to: experiments/{experiment_id}/")
        
        # Compare experiments if requested
        if args.compare_experiments:
            print("\n" + "="*60)
            print("EXPERIMENT COMPARISON")
            print("="*60)
            
            # Get all completed experiments
            experiment_dirs = [d for d in os.listdir("experiments") if os.path.isdir(f"experiments/{d}")]
            
            if len(experiment_dirs) > 1:
                comparison_df = tracker.compare_experiments(experiment_dirs)
                print(comparison_df.to_string(index=False))
                
                # Save comparison
                comparison_df.to_csv("experiment_comparison.csv", index=False)
                print(f"\nComparison saved to: experiment_comparison.csv")
            else:
                print("Only one experiment found - no comparison available")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        
        # Mark experiment as failed
        exp_dir = Path("experiments") / experiment_id
        if exp_dir.exists():
            metadata_path = exp_dir / "metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                metadata["status"] = "failed"
                metadata["error"] = str(e)
                metadata["failed_at"] = str(pd.Timestamp.now())
                
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
        
        raise e
    
    finally:
        # Force cleanup to prevent hanging
        import matplotlib
        matplotlib.pyplot.close('all')


if __name__ == "__main__":
    main()
