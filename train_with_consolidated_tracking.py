#!/usr/bin/env python3
"""
Updated training script with consolidated experiment tracking.
Integrates with post-consolidation structure and auto-archiving.
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime

# Set multiprocessing start method and disable warnings before importing other modules
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
try:
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
except:
    pass

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

# Import the new consolidated tracker
from src.utils.consolidated_experiment_tracker import consolidated_tracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Mental Health Classifier with Consolidated Tracking")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--experiment-config",
        type=str,
        help="Specific experiment configuration from experiment_configs.yaml"
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
        "--auto-archive-debug",
        action="store_true",
        help="Automatically archive debug experiments after completion"
    )
    
    parser.add_argument(
        "--check-production-promotion",
        action="store_true",
        help="Check if experiment qualifies for production promotion"
    )
    
    parser.add_argument(
        "--experiment-summary",
        action="store_true",
        help="Show experiment summary before starting"
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
    """Main training function with consolidated experiment tracking."""
    args = parse_args()
    
    # Show experiment summary if requested
    if args.experiment_summary:
        print("📊 Current Experiment Status:")
        print("=" * 50)
        summary = consolidated_tracker.get_experiment_summary()
        
        print(f"Active experiments: {summary['active_experiments']}")
        print(f"Production models: {summary['production_models']}")
        print(f"Archived (debug): {summary['archived_debug']}")
        print(f"Archived (baseline): {summary['archived_baseline']}")
        print(f"Total experiments: {summary['total_experiments']}")
        print("=" * 50)
    
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
    
    # Start experiment tracking with consolidated tracker
    experiment_id = consolidated_tracker.start_experiment(
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
        consolidated_tracker.log_metrics(experiment_id, data_stats, step=0)
        
        # Build vocabulary
        logging.info("Building vocabulary...")
        data_processor.build_vocabulary(train_texts)
        
        # Update vocab size in config
        config['model']['vocab_size'] = len(data_processor.vocab)
        logging.info(f"Vocabulary size: {config['model']['vocab_size']}")
        
        # Create data loaders with proper multiprocessing settings
        logging.info("Creating data loaders...")
        dataloaders = data_processor.create_dataloaders(
            train_texts, train_labels,
            val_texts, val_labels,
            test_texts, test_labels
        )
        
        # Override num_workers to 0 to avoid multiprocessing issues
        for dataloader_name, dataloader in dataloaders.items():
            if hasattr(dataloader, 'num_workers'):
                dataloader.num_workers = 0
        
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
        consolidated_tracker.log_metrics(experiment_id, model_stats, step=0)
        
        # Calculate class weights for imbalanced data
        class_weights = data_processor.get_class_weights(train_labels)
        logging.info(f"Class weights: {class_weights}")
        
        # Create trainer without experiment tracking parameter
        trainer = create_trainer(
            model=model,
            config=config['training'],
            device=device,
            class_weights=class_weights
        )
        
        # Train model
        logging.info("Starting training...")
        trainer.train(dataloaders['train'], dataloaders['val'])
        
        # Save model using consolidated tracker
        consolidated_tracker.save_model(
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
        model_save_dir = f"experiments/{experiment_id}/models"
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
            consolidated_tracker.log_metrics(experiment_id, test_metrics, step=-1)
        
        # Get final training metrics from trainer
        final_metrics = {
            'loss': trainer.history['val_loss'][-1] if trainer.history['val_loss'] else 0,
            'accuracy': trainer.history['val_acc'][-1] if trainer.history['val_acc'] else 0,
            'f1_macro': trainer.history['val_f1'][-1] if trainer.history['val_f1'] else 0,
            'best_val_f1': trainer.best_val_f1
        }
        
        # Finish experiment with consolidated tracker
        consolidated_tracker.finish_experiment(
            experiment_id,
            final_metrics=final_metrics,
            notes=f"Completed successfully with {config['model']['n_layer']} layers, {config['model']['n_embd']} embedding dim"
        )
        
        # Auto-archive debug experiments if requested
        if args.auto_archive_debug:
            archived_count = consolidated_tracker.cleanup_old_debug_experiments(keep_latest=2)
            if archived_count > 0:
                print(f"📦 Auto-archived {archived_count} old debug experiments")
        
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
        
        # Show experiment summary
        summary = consolidated_tracker.get_experiment_summary()
        print(f"\n📊 Updated Experiment Summary:")
        print(f"  Active experiments: {summary['active_experiments']}")
        print(f"  Production models: {summary['production_models']}")
        print(f"  Total experiments: {summary['total_experiments']}")
        
        # Check production promotion if requested
        if args.check_production_promotion:
            print("\n🏆 Production Promotion Check:")
            final_accuracy = final_metrics.get('accuracy', 0)
            final_f1 = final_metrics.get('f1_macro', 0)
            
            if final_accuracy >= 0.55 and final_f1 >= 0.45:
                print(f"✅ Qualifies for production promotion!")
                print(f"   Accuracy: {final_accuracy:.3f} (≥0.55)")
                print(f"   F1-Macro: {final_f1:.3f} (≥0.45)")
                print(f"   Run: python -c \"from src.utils.consolidated_experiment_tracker import consolidated_tracker; consolidated_tracker.promote_to_production('{experiment_id}')\"")
            else:
                print(f"❌ Does not meet production thresholds")
                print(f"   Accuracy: {final_accuracy:.3f} (need ≥0.55)")
                print(f"   F1-Macro: {final_f1:.3f} (need ≥0.45)")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        
        # Mark experiment as failed in consolidated tracker
        exp_dir = Path("experiments") / experiment_id
        if exp_dir.exists():
            metadata_path = exp_dir / "metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                metadata["status"] = "failed"
                metadata["error"] = str(e)
                metadata["failed_at"] = datetime.now().isoformat()
                
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
        
        raise e
    
    finally:
        # Enhanced cleanup to prevent multiprocessing issues
        try:
            # Close matplotlib figures
            import matplotlib.pyplot as plt
            plt.close('all')
            
            # Clean up dataloaders if they exist
            if 'dataloaders' in locals():
                del dataloaders
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Try to clean up any remaining multiprocessing resources
            try:
                import multiprocessing
                # Force cleanup of any remaining processes
                for p in multiprocessing.active_children():
                    p.terminate()
                    p.join(timeout=1)
            except:
                pass
                
        except Exception as cleanup_error:
            print(f"Warning: Cleanup error: {cleanup_error}")


if __name__ == "__main__":
    main()
