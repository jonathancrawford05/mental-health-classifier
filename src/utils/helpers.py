"""
Utility functions for mental health classifier.

Includes configuration loading, logging setup, and general helper functions.
"""

import yaml
import json
import logging
import torch
import numpy as np
import random
import os
from typing import Dict, Any, List, Optional
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup handlers
    handlers = [logging.StreamHandler()]
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get the best available device."""
    if prefer_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
    
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_model_info(model: torch.nn.Module, config: Dict[str, Any], save_path: str) -> None:
    """Save model architecture and configuration information."""
    model_info = {
        'model_architecture': str(model),
        'total_parameters': count_parameters(model),
        'trainable_parameters': count_trainable_parameters(model),
        'config': config
    }
    
    with open(save_path, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, loss: float, save_path: str, **kwargs) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    
    torch.save(checkpoint, save_path)


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, 
                 mode: str = 'min', restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            raise ValueError(f"Mode {mode} not supported. Use 'min' or 'max'.")
    
    def __call__(self, current_score: float, model: torch.nn.Module) -> bool:
        """Check if training should be stopped."""
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model)
        elif self.monitor_op(current_score, self.best_score + self.min_delta):
            self.best_score = current_score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save the current best model weights."""
        self.best_weights = model.state_dict().copy()


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def calculate_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """Calculate model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        'param_size_mb': param_size / 1024**2,
        'buffer_size_mb': buffer_size / 1024**2,
        'total_size_mb': size_all_mb
    }


def print_model_summary(model: torch.nn.Module, config: Dict[str, Any]) -> None:
    """Print comprehensive model summary."""
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    
    # Model architecture info
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    
    # Model size
    size_info = calculate_model_size(model)
    print(f"Model size: {size_info['total_size_mb']:.2f} MB")
    
    # Configuration info
    print("\nModel Configuration:")
    print("-" * 40)
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    
    print("=" * 80)


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters."""
    required_keys = [
        'model', 'training', 'data', 'labels'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate model config
    model_config = config['model']
    required_model_keys = ['vocab_size', 'n_embd', 'num_heads', 'n_layer', 'num_classes']
    
    for key in required_model_keys:
        if key not in model_config:
            raise ValueError(f"Missing required model configuration key: {key}")
    
    # Validate embedding dimension is divisible by number of heads
    if model_config['n_embd'] % model_config['num_heads'] != 0:
        raise ValueError("n_embd must be divisible by num_heads")
    
    # Validate training config
    training_config = config['training']
    required_training_keys = ['batch_size', 'learning_rate', 'num_epochs']
    
    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"Missing required training configuration key: {key}")
    
    return True


def clinical_text_stats(texts: List[str]) -> Dict[str, Any]:
    """Calculate statistics for clinical text data."""
    if not texts:
        return {}
    
    lengths = [len(text.split()) for text in texts]
    char_lengths = [len(text) for text in texts]
    
    stats = {
        'num_texts': len(texts),
        'avg_word_length': np.mean(lengths),
        'std_word_length': np.std(lengths),
        'min_word_length': min(lengths),
        'max_word_length': max(lengths),
        'median_word_length': np.median(lengths),
        'avg_char_length': np.mean(char_lengths),
        'std_char_length': np.std(char_lengths),
        'min_char_length': min(char_lengths),
        'max_char_length': max(char_lengths),
        'median_char_length': np.median(char_lengths)
    }
    
    return stats


def print_data_summary(train_texts: List[str], train_labels: List[int],
                      val_texts: List[str], val_labels: List[int],
                      test_texts: Optional[List[str]] = None,
                      test_labels: Optional[List[int]] = None,
                      label_names: List[str] = None) -> None:
    """Print comprehensive data summary."""
    print("=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    
    # Dataset sizes
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    if test_texts:
        print(f"Test samples: {len(test_texts)}")
    
    # Label distribution
    from collections import Counter
    
    train_label_counts = Counter(train_labels)
    val_label_counts = Counter(val_labels)
    
    print("\nLabel Distribution:")
    print("-" * 40)
    print("Training set:")
    for label, count in train_label_counts.items():
        label_name = label_names[label] if label_names else f"Class {label}"
        percentage = (count / len(train_labels)) * 100
        print(f"  {label_name}: {count} ({percentage:.1f}%)")
    
    print("Validation set:")
    for label, count in val_label_counts.items():
        label_name = label_names[label] if label_names else f"Class {label}"
        percentage = (count / len(val_labels)) * 100
        print(f"  {label_name}: {count} ({percentage:.1f}%)")
    
    if test_texts and test_labels:
        test_label_counts = Counter(test_labels)
        print("Test set:")
        for label, count in test_label_counts.items():
            label_name = label_names[label] if label_names else f"Class {label}"
            percentage = (count / len(test_labels)) * 100
            print(f"  {label_name}: {count} ({percentage:.1f}%)")
    
    # Text statistics
    print("\nText Statistics:")
    print("-" * 40)
    
    train_stats = clinical_text_stats(train_texts)
    print(f"Training - Avg words: {train_stats['avg_word_length']:.1f} "
          f"(±{train_stats['std_word_length']:.1f})")
    print(f"Training - Word range: {train_stats['min_word_length']} - "
          f"{train_stats['max_word_length']}")
    
    val_stats = clinical_text_stats(val_texts)
    print(f"Validation - Avg words: {val_stats['avg_word_length']:.1f} "
          f"(±{val_stats['std_word_length']:.1f})")
    print(f"Validation - Word range: {val_stats['min_word_length']} - "
          f"{val_stats['max_word_length']}")
    
    print("=" * 80)
