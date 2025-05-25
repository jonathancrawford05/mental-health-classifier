"""
Training utilities for mental health classifier.

Includes trainer class, loss functions, metrics, and training loops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import time
import os
from tqdm import tqdm
import json


class MentalHealthMetrics:
    """Metrics computation for mental health classification."""
    
    def __init__(self, label_names: List[str]):
        self.label_names = label_names
        self.num_classes = len(label_names)
    
    def compute_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive metrics for classification."""
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro, 
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }
        
        # Per-class metrics
        for i, label_name in enumerate(self.label_names):
            metrics[f'precision_{label_name.lower()}'] = precision[i]
            metrics[f'recall_{label_name.lower()}'] = recall[i]
            metrics[f'f1_{label_name.lower()}'] = f1[i]
            metrics[f'support_{label_name.lower()}'] = support[i]
        
        return metrics
    
    def plot_confusion_matrix(self, predictions: np.ndarray, labels: np.ndarray, 
                            save_path: Optional[str] = None) -> None:
        """Plot and optionally save confusion matrix."""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_names, yticklabels=self.label_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_classification_report(self, predictions: np.ndarray, labels: np.ndarray) -> None:
        """Print detailed classification report."""
        report = classification_report(labels, predictions, target_names=self.label_names)
        print("Classification Report:")
        print("=" * 50)
        print(report)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in mental health classification."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class Trainer:
    """Training class for mental health classifier."""
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict,
                 device: torch.device,
                 class_weights: Optional[torch.Tensor] = None):
        
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Initialize loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            # Use focal loss for imbalanced classes
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 2e-5),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Initialize scheduler
        self.scheduler = None
        
        # Initialize metrics
        label_names = config.get('label_names', ['Depression', 'Anxiety', 'Suicide'])
        self.metrics = MentalHealthMetrics(label_names)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }
        
        self.best_val_f1 = 0.0
        self.best_model_state = None
    
    def setup_scheduler(self, train_dataloader: DataLoader) -> None:
        """Setup learning rate scheduler."""
        total_steps = len(train_dataloader) * self.config.get('num_epochs', 10)
        warmup_steps = self.config.get('warmup_steps', total_steps // 10)
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-7
        )
    
    def train_epoch(self, train_dataloader: DataLoader) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs['logits'], labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip_norm']
                )
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs['logits'], dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_dataloader)
        metrics = self.metrics.compute_metrics(
            np.array(all_predictions), 
            np.array(all_labels)
        )
        
        return avg_loss, metrics['accuracy'], metrics['f1_macro']
    
    def validate_epoch(self, val_dataloader: DataLoader) -> Tuple[float, float, float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs['logits'], labels)
                
                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(outputs['logits'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(val_dataloader)
        metrics = self.metrics.compute_metrics(
            np.array(all_predictions), 
            np.array(all_labels)
        )
        
        return avg_loss, metrics['accuracy'], metrics['f1_macro'], metrics
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        """Main training loop."""
        logging.info("Starting training...")
        
        # Setup scheduler
        self.setup_scheduler(train_dataloader)
        
        num_epochs = self.config.get('num_epochs', 10)
        save_dir = self.config.get('model_save_dir', 'models/')
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc, train_f1 = self.train_epoch(train_dataloader)
            
            # Validation phase
            val_loss, val_acc, val_f1, val_metrics = self.validate_epoch(val_dataloader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_f1': self.best_val_f1,
                    'config': self.config
                }
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
                logging.info(f"New best model saved with val F1: {val_f1:.4f}")
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            logging.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f} - "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Print detailed validation metrics
            if (epoch + 1) % 5 == 0:
                print("\nDetailed Validation Metrics:")
                for metric_name, value in val_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"{metric_name}: {value:.4f}")
        
        logging.info(f"Training completed. Best validation F1: {self.best_val_f1:.4f}")
    
    def evaluate(self, test_dataloader: DataLoader, 
                save_plots: bool = True, save_dir: str = 'results/') -> Dict[str, float]:
        """Evaluate model on test set."""
        logging.info("Starting evaluation...")
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                probabilities = F.softmax(outputs['logits'], dim=-1)
                predictions = torch.argmax(outputs['logits'], dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        predictions_array = np.array(all_predictions)
        labels_array = np.array(all_labels)
        
        metrics = self.metrics.compute_metrics(predictions_array, labels_array)
        
        # Print results
        self.metrics.print_classification_report(predictions_array, labels_array)
        
        if save_plots:
            os.makedirs(save_dir, exist_ok=True)
            
            # Plot confusion matrix
            self.metrics.plot_confusion_matrix(
                predictions_array, labels_array,
                save_path=os.path.join(save_dir, 'confusion_matrix.png')
            )
            
            # Plot training history
            self.plot_training_history(save_path=os.path.join(save_dir, 'training_history.png'))
        
        return metrics
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train Accuracy')
        axes[0, 1].plot(self.history['val_acc'], label='Val Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score plot
        axes[1, 0].plot(self.history['train_f1'], label='Train F1')
        axes[1, 0].plot(self.history['val_f1'], label='Val F1')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot (if scheduler is used)
        if self.scheduler:
            lrs = []
            for _ in range(len(self.history['train_loss'])):
                lrs.append(self.scheduler.get_last_lr()[0])
            axes[1, 1].plot(lrs)
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_training_history(self, save_path: str) -> None:
        """Save training history to JSON."""
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logging.info(f"Training history saved to {save_path}")
    
    def predict_text(self, text: str, data_processor, return_probabilities: bool = False):
        """Predict class for a single text input."""
        self.model.eval()
        
        # Preprocess text
        preprocessed_text = data_processor.preprocessor.preprocess(text)
        tokens = data_processor.tokenizer(preprocessed_text)
        token_ids = [data_processor.vocab[token] for token in tokens]
        
        # Use the data processor's max_length which should match model's max_seq_length
        max_length = data_processor.config.get('max_length', 512)
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        # Pad to max_length
        pad_token_id = data_processor.vocab['<pad>']
        padding_length = max_length - len(token_ids)
        token_ids.extend([pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        
        # Convert to tensors
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = F.softmax(outputs['logits'], dim=-1)
            prediction = torch.argmax(outputs['logits'], dim=-1)
        
        predicted_class = data_processor.decode_labels([prediction.item()])[0]
        
        if return_probabilities:
            probs = probabilities[0].cpu().numpy()
            prob_dict = {}
            for i, label in enumerate(data_processor.label_names):
                prob_dict[label] = float(probs[i])
            return predicted_class, prob_dict
        
        return predicted_class


def create_trainer(model: nn.Module, config: Dict, device: torch.device,
                  class_weights: Optional[torch.Tensor] = None) -> Trainer:
    """Factory function to create trainer."""
    return Trainer(model, config, device, class_weights)
