"""
Comprehensive experiment tracking and model versioning system.
Integrates with MLflow for advanced tracking capabilities.
"""

import json
import os
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib
import torch
import pandas as pd

class ExperimentTracker:
    """
    Comprehensive experiment tracking system for model development.
    """
    
    def __init__(self, project_name: str = "mental-health-classifier", base_dir: str = "."):
        self.project_name = project_name
        self.base_dir = Path(base_dir)
        self.experiments_dir = self.base_dir / "experiments"
        self.experiments_dir.mkdir(exist_ok=True)
        
        # Create tracking files
        self.experiments_log = self.base_dir / "experiments_log.jsonl"
        self.models_registry = self.base_dir / "models_registry.json"
        
        # Initialize registry if it doesn't exist
        if not self.models_registry.exists():
            self._init_models_registry()
    
    def _init_models_registry(self):
        """Initialize the models registry."""
        registry = {
            "project_name": self.project_name,
            "created_at": datetime.now().isoformat(),
            "models": {},
            "best_models": {
                "highest_accuracy": None,
                "best_f1_macro": None,
                "best_suicide_recall": None,
                "most_stable": None
            }
        }
        
        with open(self.models_registry, "w") as f:
            json.dump(registry, f, indent=2)
    
    def start_experiment(self, 
                        experiment_name: str,
                        description: str,
                        config: Dict,
                        tags: List[str] = None) -> str:
        """
        Start a new experiment and return experiment ID.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"
        
        # Create experiment directory
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (exp_dir / "config").mkdir(exist_ok=True)
        (exp_dir / "models").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "results").mkdir(exist_ok=True)
        (exp_dir / "plots").mkdir(exist_ok=True)
        
        # Save experiment metadata
        experiment_metadata = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "description": description,
            "config": config,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "status": "running",
            "metrics": {},
            "artifacts": [],
            "git_commit": self._get_git_commit(),
            "config_hash": self._hash_config(config)
        }
        
        # Save metadata
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(experiment_metadata, f, indent=2)
        
        # Save config
        with open(exp_dir / "config" / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Log experiment start
        self._log_experiment(experiment_metadata)
        
        print(f"🚀 Started experiment: {experiment_id}")
        return experiment_id
    
    def log_metrics(self, 
                   experiment_id: str, 
                   metrics: Dict[str, float],
                   step: Optional[int] = None,
                   epoch: Optional[int] = None):
        """Log metrics for an experiment."""
        
        exp_dir = self.experiments_dir / experiment_id
        if not exp_dir.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Load current metadata
        with open(exp_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Update metrics
        timestamp = datetime.now().isoformat()
        metric_entry = {
            "timestamp": timestamp,
            "step": step,
            "epoch": epoch,
            "metrics": metrics
        }
        
        # Initialize metrics list if not exists
        if "metric_history" not in metadata:
            metadata["metric_history"] = []
        
        metadata["metric_history"].append(metric_entry)
        
        # Update best metrics
        if "best_metrics" not in metadata:
            metadata["best_metrics"] = {}
        
        for metric_name, value in metrics.items():
            if metric_name not in metadata["best_metrics"]:
                metadata["best_metrics"][metric_name] = {"value": value, "step": step, "epoch": epoch}
            else:
                # For accuracy, f1, recall - higher is better
                # For loss - lower is better
                if "loss" in metric_name.lower():
                    if value < metadata["best_metrics"][metric_name]["value"]:
                        metadata["best_metrics"][metric_name] = {"value": value, "step": step, "epoch": epoch}
                else:
                    if value > metadata["best_metrics"][metric_name]["value"]:
                        metadata["best_metrics"][metric_name] = {"value": value, "step": step, "epoch": epoch}
        
        # Save updated metadata
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Also save metrics to CSV for easy analysis
        self._save_metrics_csv(exp_dir, metadata["metric_history"])
    
    def save_model(self, 
                  experiment_id: str,
                  model: torch.nn.Module,
                  model_name: str = "best_model",
                  additional_info: Dict = None):
        """Save a model checkpoint."""
        
        exp_dir = self.experiments_dir / experiment_id
        if not exp_dir.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        model_path = exp_dir / "models" / f"{model_name}.pth"
        
        # Save model state
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_architecture': str(model),
            'saved_at': datetime.now().isoformat(),
            'additional_info': additional_info or {}
        }, model_path)
        
        # Update experiment metadata
        with open(exp_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        if "saved_models" not in metadata:
            metadata["saved_models"] = []
        
        metadata["saved_models"].append({
            "model_name": model_name,
            "model_path": str(model_path),
            "saved_at": datetime.now().isoformat(),
            "additional_info": additional_info or {}
        })
        
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Model saved: {model_path}")
        return model_path
    
    def finish_experiment(self, 
                         experiment_id: str,
                         final_metrics: Dict[str, float] = None,
                         notes: str = None):
        """Mark experiment as completed."""
        
        exp_dir = self.experiments_dir / experiment_id
        if not exp_dir.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Load and update metadata
        with open(exp_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        metadata["status"] = "completed"
        metadata["completed_at"] = datetime.now().isoformat()
        metadata["final_metrics"] = final_metrics or {}
        metadata["notes"] = notes or ""
        
        # Calculate total training time
        created_at = datetime.fromisoformat(metadata["created_at"])
        completed_at = datetime.now()
        metadata["total_duration_seconds"] = (completed_at - created_at).total_seconds()
        
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Update models registry
        self._update_models_registry(experiment_id, metadata)
        
        print(f"✅ Experiment completed: {experiment_id}")
        print(f"Duration: {metadata['total_duration_seconds']/60:.1f} minutes")
    
    def compare_experiments(self, 
                           experiment_ids: List[str],
                           metrics: List[str] = None) -> pd.DataFrame:
        """Compare multiple experiments."""
        
        if metrics is None:
            metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
        
        results = []
        
        for exp_id in experiment_ids:
            exp_dir = self.experiments_dir / exp_id
            if not exp_dir.exists():
                print(f"⚠️ Experiment {exp_id} not found")
                continue
            
            with open(exp_dir / "metadata.json", "r") as f:
                metadata = json.load(f)
            
            result = {
                "experiment_id": exp_id,
                "experiment_name": metadata.get("experiment_name", ""),
                "description": metadata.get("description", ""),
                "status": metadata.get("status", ""),
                "duration_minutes": metadata.get("total_duration_seconds", 0) / 60
            }
            
            # Add best metrics
            best_metrics = metadata.get("best_metrics", {})
            for metric in metrics:
                result[f"best_{metric}"] = best_metrics.get(metric, {}).get("value", None)
            
            # Add final metrics
            final_metrics = metadata.get("final_metrics", {})
            for metric in metrics:
                result[f"final_{metric}"] = final_metrics.get(metric, None)
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_best_models(self) -> Dict:
        """Get best performing models across all experiments."""
        
        with open(self.models_registry, "r") as f:
            registry = json.load(f)
        
        return registry.get("best_models", {})
    
    def _log_experiment(self, metadata: Dict):
        """Log experiment to JSONL file."""
        
        log_entry = {
            "experiment_id": metadata["experiment_id"],
            "timestamp": metadata["created_at"],
            "experiment_name": metadata["experiment_name"],
            "description": metadata["description"],
            "config_hash": metadata["config_hash"],
            "tags": metadata["tags"],
            "status": metadata["status"]
        }
        
        with open(self.experiments_log, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def _save_metrics_csv(self, exp_dir: Path, metric_history: List[Dict]):
        """Save metrics history to CSV."""
        
        if not metric_history:
            return
        
        # Flatten metrics history
        rows = []
        for entry in metric_history:
            row = {
                "timestamp": entry["timestamp"],
                "step": entry.get("step"),
                "epoch": entry.get("epoch")
            }
            row.update(entry["metrics"])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(exp_dir / "results" / "metrics_history.csv", index=False)
    
    def _update_models_registry(self, experiment_id: str, metadata: Dict):
        """Update the global models registry."""
        
        with open(self.models_registry, "r") as f:
            registry = json.load(f)
        
        # Add this experiment to registry
        registry["models"][experiment_id] = {
            "experiment_name": metadata.get("experiment_name"),
            "final_metrics": metadata.get("final_metrics", {}),
            "best_metrics": metadata.get("best_metrics", {}),
            "completed_at": metadata.get("completed_at"),
            "config_hash": metadata.get("config_hash"),
            "status": metadata.get("status")
        }
        
        # Update best models
        final_metrics = metadata.get("final_metrics", {})
        best_metrics = metadata.get("best_metrics", {})
        
        # Check if this is the best model for various metrics
        current_best = registry["best_models"]
        
        # Highest accuracy
        accuracy = final_metrics.get("accuracy") or best_metrics.get("accuracy", {}).get("value")
        if accuracy and (not current_best["highest_accuracy"] or 
                        accuracy > registry["models"][current_best["highest_accuracy"]]["final_metrics"].get("accuracy", 0)):
            current_best["highest_accuracy"] = experiment_id
        
        # Best F1 macro
        f1_macro = final_metrics.get("f1_macro") or best_metrics.get("f1_macro", {}).get("value")
        if f1_macro and (not current_best["best_f1_macro"] or 
                        f1_macro > registry["models"][current_best["best_f1_macro"]]["final_metrics"].get("f1_macro", 0)):
            current_best["best_f1_macro"] = experiment_id
        
        # Best suicide recall (most important clinically)
        suicide_recall = final_metrics.get("recall_class_2") or best_metrics.get("recall_class_2", {}).get("value")
        if suicide_recall and (not current_best["best_suicide_recall"] or 
                              suicide_recall > registry["models"][current_best["best_suicide_recall"]]["final_metrics"].get("recall_class_2", 0)):
            current_best["best_suicide_recall"] = experiment_id
        
        registry["best_models"] = current_best
        
        with open(self.models_registry, "w") as f:
            json.dump(registry, f, indent=2)
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import git
            repo = git.Repo(".")
            return repo.head.commit.hexsha
        except:
            return None
    
    def _hash_config(self, config: Dict) -> str:
        """Create hash of configuration for tracking."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

# Global tracker instance
tracker = ExperimentTracker()
