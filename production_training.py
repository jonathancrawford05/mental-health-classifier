#!/usr/bin/env python3
"""
Production-ready training script based on the proven working approach.
Scales up the successful ultra-minimal method with full features.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import json
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Proven environment setup
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

torch.set_num_threads(1)

# Setup paths
os.chdir("/Users/family_crawfords/projects/claude-mcp/mental-health-classifier")
sys.path.insert(0, "src")

def parse_args():
    parser = argparse.ArgumentParser(description="Production Mental Health Classifier Training")
    parser.add_argument("--experiment-config", type=str, help="Experiment configuration")
    parser.add_argument("--experiment-name", type=str, help="Custom experiment name")
    parser.add_argument("--description", type=str, help="Experiment description")
    parser.add_argument("--create-sample-data", action="store_true", help="Create sample data")
    parser.add_argument("--sample-size", type=int, default=1000, help="Sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cpu", help="Device")
    return parser.parse_args()

class ProductionTracker:
    """Production-ready experiment tracker based on proven approach."""
    
    def __init__(self):
        self.experiments_dir = Path("experiments")
        self.experiments_dir.mkdir(exist_ok=True)
    
    def start_experiment(self, name: str, description: str, config: dict) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{name}_{timestamp}"
        
        # Create directory structure
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        (exp_dir / "models").mkdir(exist_ok=True)
        (exp_dir / "results").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        
        # Save metadata
        metadata = {
            "experiment_id": experiment_id,
            "name": name,
            "description": description,
            "config": config,
            "created_at": datetime.now().isoformat(),
            "status": "running"
        }
        
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"🧪 Experiment started: {experiment_id}")
        return experiment_id
    
    def log_metrics(self, experiment_id: str, metrics: dict, epoch: int):
        exp_dir = self.experiments_dir / experiment_id
        
        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            **metrics
        }
        
        # Append to metrics log
        with open(exp_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(metric_entry) + "\n")
    
    def save_model(self, experiment_id: str, model: nn.Module, name: str, extra_info: dict = None):
        exp_dir = self.experiments_dir / experiment_id
        model_path = exp_dir / "models" / f"{name}.pth"
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'saved_at': datetime.now().isoformat()
        }
        
        if extra_info:
            save_dict.update(extra_info)
        
        torch.save(save_dict, model_path)
        print(f"💾 Model saved: {model_path}")
        return model_path
    
    def save_results(self, experiment_id: str, results: dict):
        exp_dir = self.experiments_dir / experiment_id
        
        with open(exp_dir / "results" / "final_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    def finish_experiment(self, experiment_id: str, final_metrics: dict = None):
        exp_dir = self.experiments_dir / experiment_id
        
        # Update metadata
        with open(exp_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        metadata.update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "final_metrics": final_metrics or {}
        })
        
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Experiment completed: {experiment_id}")

def evaluate_model(model, dataloader, device, label_names):
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs['logits'], labels)
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs['logits'], dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }
    
    # Per-class metrics
    for i, label_name in enumerate(label_names):
        metrics[f'precision_{label_name.lower()}'] = precision[i] if i < len(precision) else 0.0
        metrics[f'recall_{label_name.lower()}'] = recall[i] if i < len(recall) else 0.0
        metrics[f'f1_{label_name.lower()}'] = f1[i] if i < len(f1) else 0.0
    
    return metrics, all_predictions, all_labels

def train_model(model, train_loader, val_loader, config, device, tracker, experiment_id, label_names):
    """Production training function based on proven approach."""
    
    # Setup optimizer (using proven simple approach)
    learning_rate = config['training'].get('learning_rate', 0.001)
    weight_decay = config['training'].get('weight_decay', 0.01)
    
    # Use Adam instead of SGD for better convergence
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = config['training'].get('num_epochs', 10)
    gradient_clip = config['training'].get('gradient_clip_norm', 1.0)
    
    print(f"🚀 Starting training for {num_epochs} epochs...")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Gradient clipping: {gradient_clip}")
    
    best_val_f1 = 0.0
    best_model_state = None
    training_history = []
    
    for epoch in range(num_epochs):
        print(f"\n📊 Epoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs['logits'], labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Progress updates
            if (batch_idx + 1) % 20 == 0:
                print(f"    Batch {batch_idx + 1}/{len(train_loader)}: Loss = {loss.item():.4f}")
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        print("  Validating...")
        val_metrics, _, _ = evaluate_model(model, val_loader, device, label_names)
        
        # Store epoch results
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_f1_macro': val_metrics['f1_macro'],
            'val_precision_macro': val_metrics['precision_macro'],
            'val_recall_macro': val_metrics['recall_macro']
        }
        
        training_history.append(epoch_results)
        
        # Log metrics
        tracker.log_metrics(experiment_id, epoch_results, epoch + 1)
        
        # Print results
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1 Macro: {val_metrics['f1_macro']:.4f}")
        
        # Save best model
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_model_state = model.state_dict().copy()
            
            tracker.save_model(experiment_id, model, "best_model", {
                'epoch': epoch + 1,
                'val_f1_macro': best_val_f1,
                'config': config
            })
            
            print(f"  💾 New best model saved (F1: {best_val_f1:.4f})")
    
    # Load best model for final evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\n✅ Training completed! Best F1: {best_val_f1:.4f}")
    return model, training_history

def main():
    args = parse_args()
    
    print("🚀 PRODUCTION MENTAL HEALTH CLASSIFIER TRAINING")
    print("=" * 60)
    print("Using the proven working approach, scaled for production")
    print("=" * 60)
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load configuration
    from utils import load_config
    config = load_config("config/config.yaml")
    
    if args.experiment_config:
        with open("config/experiment_configs.yaml", 'r') as f:
            experiment_configs = yaml.safe_load(f)
        
        if args.experiment_config not in experiment_configs:
            available = list(experiment_configs.keys())
            raise ValueError(f"Config '{args.experiment_config}' not found. Available: {available}")
        
        exp_config = experiment_configs[args.experiment_config]
        
        # Merge configurations
        import copy
        config = copy.deepcopy(config)
        
        def deep_merge(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value
        
        deep_merge(config, exp_config)
        
        experiment_name = args.experiment_name or args.experiment_config
        description = args.description or exp_config.get('description', f"Training with {args.experiment_config}")
    else:
        experiment_name = args.experiment_name or "production_training"
        description = args.description or "Production training run"
    
    print(f"📋 Experiment: {experiment_name}")
    print(f"📝 Description: {description}")
    
    # Setup device
    device = torch.device(args.device)
    print(f"🖥️  Device: {device}")
    
    # Initialize tracker
    tracker = ProductionTracker()
    experiment_id = tracker.start_experiment(experiment_name, description, config)
    
    try:
        # Create sample data if requested
        if args.create_sample_data:
            print(f"\n📊 Creating sample dataset ({args.sample_size} samples)...")
            
            from data import create_sample_data
            
            os.makedirs("data", exist_ok=True)
            create_sample_data("data/sample_train.csv", args.sample_size)
            
            # Create splits
            df = pd.read_csv("data/sample_train.csv")
            
            train_val_df, test_df = train_test_split(
                df, test_size=0.2, random_state=args.seed, stratify=df['label']
            )
            
            train_df, val_df = train_test_split(
                train_val_df, test_size=0.25, random_state=args.seed, stratify=train_val_df['label']
            )
            
            # Save splits
            train_df.to_csv("data/train.csv", index=False)
            val_df.to_csv("data/val.csv", index=False)
            test_df.to_csv("data/test.csv", index=False)
            
            print("✅ Sample data created and split")
            
            # Update config paths
            config['data']['train_path'] = "data/train.csv"
            config['data']['val_path'] = "data/val.csv"
            config['data']['test_path'] = "data/test.csv"
        
        # Process data using proven approach
        print("\n📝 Processing data...")
        from data import DataProcessor
        
        config['data']['max_length'] = config['model']['max_seq_length']
        config['data']['batch_size'] = config['training']['batch_size']
        data_processor = DataProcessor(config['data'])
        
        # Load data
        train_texts, train_labels = data_processor.load_data(config['data']['train_path'])
        val_texts, val_labels = data_processor.load_data(config['data']['val_path'])
        
        test_texts, test_labels = None, None
        if 'test_path' in config['data'] and os.path.exists(config['data']['test_path']):
            test_texts, test_labels = data_processor.load_data(config['data']['test_path'])
        
        print(f"  Training samples: {len(train_texts)}")
        print(f"  Validation samples: {len(val_texts)}")
        if test_texts:
            print(f"  Test samples: {len(test_texts)}")
        
        # Build vocabulary
        data_processor.build_vocabulary(train_texts)
        config['model']['vocab_size'] = len(data_processor.vocab)
        print(f"  Vocabulary size: {config['model']['vocab_size']}")
        
        # Create dataloaders
        dataloaders = data_processor.create_dataloaders(
            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
        )
        
        print("✅ Data processing completed")
        
        # Create model using proven approach
        print("\n🧠 Creating model...")
        from models import create_model
        
        model = create_model(config['model']).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {param_count:,}")
        
        # Display model configuration
        print(f"  Architecture: {config['model']['name']}")
        print(f"  Embedding dim: {config['model']['n_embd']}")
        print(f"  Layers: {config['model']['n_layer']}")
        print(f"  Attention heads: {config['model']['num_heads']}")
        print(f"  Max sequence length: {config['model']['max_seq_length']}")
        
        # Train model
        label_names = config['labels']['label_names']
        trained_model, history = train_model(
            model, dataloaders['train'], dataloaders['val'], 
            config, device, tracker, experiment_id, label_names
        )
        
        # Save final model
        tracker.save_model(experiment_id, trained_model, "final_model", {
            'config': config,
            'vocab_size': len(data_processor.vocab),
            'label_names': label_names
        })
        
        # Save vocabulary
        exp_dir = Path("experiments") / experiment_id
        vocab_path = exp_dir / "models" / "vocab.pkl"
        data_processor.save_vocabulary(str(vocab_path))
        print(f"💾 Vocabulary saved: {vocab_path}")
        
        # Final evaluation on test set
        final_metrics = {}
        if test_texts:
            print("\n📊 Final evaluation on test set...")
            test_metrics, test_predictions, test_labels_array = evaluate_model(
                trained_model, dataloaders['test'], device, label_names
            )
            
            final_metrics = test_metrics
            
            print("Test Results:")
            print(f"  Test Loss: {test_metrics['loss']:.4f}")
            print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  Test F1 Macro: {test_metrics['f1_macro']:.4f}")
            
            # Generate classification report
            report = classification_report(
                test_labels_array, test_predictions, 
                target_names=label_names, output_dict=True
            )
            
            print("\nDetailed Classification Report:")
            for label in label_names:
                if label.lower() in report:
                    metrics = report[label.lower()]
                    print(f"  {label}:")
                    print(f"    Precision: {metrics['precision']:.4f}")
                    print(f"    Recall: {metrics['recall']:.4f}")
                    print(f"    F1-score: {metrics['f1-score']:.4f}")
        
        # Save comprehensive results
        results = {
            'experiment_id': experiment_id,
            'config': config,
            'model_parameters': param_count,
            'training_history': history,
            'final_metrics': final_metrics,
            'completed_at': datetime.now().isoformat()
        }
        
        tracker.save_results(experiment_id, results)
        tracker.finish_experiment(experiment_id, final_metrics)
        
        print("\n🎉 SUCCESS!")
        print("=" * 60)
        print(f"Experiment ID: {experiment_id}")
        print(f"Results saved to: experiments/{experiment_id}/")
        print(f"Model parameters: {param_count:,}")
        if final_metrics:
            print(f"Final test accuracy: {final_metrics['accuracy']:.4f}")
            print(f"Final test F1: {final_metrics['f1_macro']:.4f}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Mark experiment as failed
        try:
            tracker.finish_experiment(experiment_id, {"error": str(e), "status": "failed"})
        except:
            pass
        
        raise e

if __name__ == "__main__":
    main()
