"""
Consolidated Experiment Tracker - Post-Consolidation Version
Enhanced to work with archived experiments and production model structure.
"""

import json
import os
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import torch
import pandas as pd

class ConsolidatedExperimentTracker:
    """
    Enhanced experiment tracking that integrates with consolidation structure.
    Automatically handles archiving and production model promotion.
    """
    
    def __init__(self, project_name: str = "mental-health-classifier", base_dir: str = "."):
        self.project_name = project_name
        self.base_dir = Path(base_dir)
        
        # Directory structure
        self.experiments_dir = self.base_dir / "experiments"
        self.production_dir = self.base_dir / "models" / "production"
        self.archive_debug_dir = self.experiments_dir / "archive_debug"
        self.archive_baseline_dir = self.experiments_dir / "archive_baseline"
        
        # Create directories
        for dir_path in [self.experiments_dir, self.production_dir, 
                        self.archive_debug_dir, self.archive_baseline_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Tracking files
        self.experiments_log = self.base_dir / "experiments_log.jsonl"
        self.models_registry = self.base_dir / "models_registry_consolidated.json"
        self.consolidation_log = self.base_dir / "experiment_consolidation_log.json"
        
        # Initialize tracking files if they don't exist
        if not self.models_registry.exists():
            self._init_consolidated_registry()
        
        self._load_consolidation_config()
    
    def _init_consolidated_registry(self):
        """Initialize the consolidated models registry."""
        registry = {
            "project_name": self.project_name,
            "created_at": datetime.now().isoformat(),
            "consolidation_version": "v2.0",
            "last_updated": datetime.now().isoformat(),
            
            "production_models": {},
            "active_experiments": {},
            "archived_experiments": {
                "archive_debug": {},
                "archive_baseline": {},
                "archived": {}
            },
            
            "model_recommendations": {
                "production_deployment": None,
                "high_capacity_experiments": None,
                "debugging_reference": None
            },
            
            "auto_archival_rules": {
                "debug_experiments": {
                    "enabled": True,
                    "criteria": ["debug", "test", "mock", "ultra-micro", "micro"],
                    "min_experiments_before_archive": 3,
                    "archive_location": "archive_debug"
                },
                "failed_experiments": {
                    "enabled": True,
                    "auto_archive_after_days": 7,
                    "archive_location": "archive_baseline"
                },
                "production_promotion": {
                    "enabled": True,
                    "min_accuracy_threshold": 0.55,
                    "min_f1_macro_threshold": 0.45,
                    "require_manual_approval": True
                }
            },
            
            "tracking_metadata": {
                "total_experiments_run": 0,
                "experiments_promoted_to_production": 0,
                "experiments_auto_archived": 0
            }
        }
        
        with open(self.models_registry, "w") as f:
            json.dump(registry, f, indent=2)
    
    def _load_consolidation_config(self):
        """Load consolidation configuration if available."""
        try:
            if self.consolidation_log.exists():
                with open(self.consolidation_log, "r") as f:
                    self.consolidation_config = json.load(f)
            else:
                self.consolidation_config = {}
        except Exception as e:
            print(f"⚠️ Warning: Could not load consolidation config: {e}")
            self.consolidation_config = {}
    
    def start_experiment(self, 
                        experiment_name: str,
                        description: str,
                        config: Dict,
                        tags: List[str] = None) -> str:
        """
        Start a new experiment with enhanced tracking and auto-classification.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"
        
        # Classify experiment type for potential auto-archiving
        exp_category = self._classify_experiment(experiment_name, description, tags or [])
        
        # Create experiment directory
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        for subdir in ["config", "models", "logs", "results", "plots"]:
            (exp_dir / subdir).mkdir(exist_ok=True)
        
        # Enhanced experiment metadata
        experiment_metadata = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "description": description,
            "config": config,
            "tags": tags or [],
            "experiment_category": exp_category,
            "created_at": datetime.now().isoformat(),
            "status": "running",
            "consolidation_status": "active",
            "metrics": {},
            "artifacts": [],
            "git_commit": self._get_git_commit(),
            "config_hash": self._hash_config(config),
            "tracking_version": "v2.0_consolidated"
        }
        
        # Save metadata
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(experiment_metadata, f, indent=2)
        
        # Save config
        with open(exp_dir / "config" / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Log experiment start
        self._log_experiment(experiment_metadata)
        
        # Update registry
        self._update_active_experiments(experiment_id, experiment_metadata)
        
        print(f"🚀 Started experiment: {experiment_id}")
        print(f"📂 Category: {exp_category}")
        
        return experiment_id
    
    def finish_experiment(self, 
                         experiment_id: str,
                         final_metrics: Dict[str, float] = None,
                         notes: str = None):
        """
        Finish experiment with auto-archiving and production promotion checks.
        """
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
        
        # Check for production promotion
        promotion_result = self._check_production_promotion(experiment_id, metadata)
        
        # Check for auto-archiving
        archival_result = self._check_auto_archival(experiment_id, metadata)
        
        # Update consolidated registry
        self._update_consolidated_registry(experiment_id, metadata, promotion_result, archival_result)
        
        print(f"✅ Experiment completed: {experiment_id}")
        print(f"⏱️ Duration: {metadata['total_duration_seconds']/60:.1f} minutes")
        
        if promotion_result["should_promote"]:
            print(f"🏆 {promotion_result['message']}")
        
        if archival_result["should_archive"]:
            print(f"📦 {archival_result['message']}")
    
    def _classify_experiment(self, name: str, description: str, tags: List[str]) -> str:
        """Classify experiment for auto-archiving rules."""
        
        debug_keywords = ["debug", "test", "mock", "ultra-micro", "micro", "minimal", "tiny"]
        baseline_keywords = ["baseline", "production", "medium", "large", "scaled"]
        
        name_lower = name.lower()
        desc_lower = description.lower()
        tags_lower = [tag.lower() for tag in tags]
        
        # Check for debug experiments
        if any(keyword in name_lower or keyword in desc_lower for keyword in debug_keywords):
            return "debug"
        
        if any(keyword in tags_lower for keyword in debug_keywords):
            return "debug"
        
        # Check for baseline experiments  
        if any(keyword in name_lower or keyword in desc_lower for keyword in baseline_keywords):
            return "baseline"
            
        if any(keyword in tags_lower for keyword in baseline_keywords):
            return "baseline"
        
        # Default to baseline for regular experiments
        return "baseline"
    
    def _check_production_promotion(self, experiment_id: str, metadata: Dict) -> Dict:
        """Check if experiment should be promoted to production."""
        
        try:
            with open(self.models_registry, "r") as f:
                registry = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If registry doesn't exist or is corrupted, return no promotion
            return {"should_promote": False, "reason": "registry_unavailable"}
        
        # Get promotion rules with defaults if not present
        promotion_rules = registry.get("auto_archival_rules", {}).get("production_promotion", {
            "enabled": True,
            "min_accuracy_threshold": 0.55,
            "min_f1_macro_threshold": 0.45,
            "require_manual_approval": True
        })
        
        if not promotion_rules.get("enabled", True):
            return {"should_promote": False, "reason": "auto_promotion_disabled"}
        
        final_metrics = metadata.get("final_metrics", {})
        
        # Check thresholds
        accuracy = final_metrics.get("accuracy", 0)
        f1_macro = final_metrics.get("f1_macro", 0)
        
        meets_accuracy = accuracy >= promotion_rules.get("min_accuracy_threshold", 0.55)
        meets_f1 = f1_macro >= promotion_rules.get("min_f1_macro_threshold", 0.45)
        
        if meets_accuracy and meets_f1:
            message = f"Candidate for production promotion (Acc: {accuracy:.3f}, F1: {f1_macro:.3f})"
            
            if promotion_rules.get("require_manual_approval", True):
                return {
                    "should_promote": False,
                    "reason": "requires_manual_approval",
                    "message": message + " - Manual approval required",
                    "promotion_candidate": True
                }
            else:
                return {
                    "should_promote": True,
                    "reason": "meets_thresholds",
                    "message": message + " - Auto-promoting to production",
                    "promotion_candidate": True
                }
        
        return {
            "should_promote": False,
            "reason": "below_thresholds",
            "message": f"Below promotion thresholds (Acc: {accuracy:.3f}/{promotion_rules.get('min_accuracy_threshold', 0.55)}, F1: {f1_macro:.3f}/{promotion_rules.get('min_f1_macro_threshold', 0.45)})"
        }
    
    def _check_auto_archival(self, experiment_id: str, metadata: Dict) -> Dict:
        """Check if experiment should be auto-archived."""
        
        try:
            with open(self.models_registry, "r") as f:
                registry = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If registry doesn't exist, return no archival
            return {"should_archive": False, "reason": "registry_unavailable"}
        
        # Get archival rules with defaults if not present
        archival_rules = registry.get("auto_archival_rules", {
            "debug_experiments": {
                "enabled": True,
                "criteria": ["debug", "test", "mock", "ultra-micro", "micro"],
                "min_experiments_before_archive": 3,
                "archive_location": "archive_debug"
            },
            "failed_experiments": {
                "enabled": True,
                "auto_archive_after_days": 7,
                "archive_location": "archive_baseline"
            }
        })
        
        exp_category = metadata.get("experiment_category", "baseline")
        
        # Check debug experiments
        if exp_category == "debug" and archival_rules.get("debug_experiments", {}).get("enabled", True):
            debug_rules = archival_rules["debug_experiments"]
            debug_count = len([d for d in os.listdir(self.experiments_dir) 
                             if d.startswith("baseline_micro") or d.startswith("baseline_ultra_micro")])
            
            if debug_count >= debug_rules.get("min_experiments_before_archive", 3):
                return {
                    "should_archive": True,
                    "archive_location": debug_rules.get("archive_location", "archive_debug"),
                    "reason": "debug_experiment_limit",
                    "message": f"Auto-archiving debug experiment ({debug_count} debug experiments found)"
                }
        
        # Check failed experiments
        if metadata.get("status") == "failed" and archival_rules.get("failed_experiments", {}).get("enabled", True):
            failed_rules = archival_rules["failed_experiments"]
            return {
                "should_archive": True,
                "archive_location": failed_rules.get("archive_location", "archive_baseline"),
                "reason": "failed_experiment",
                "message": "Auto-archiving failed experiment"
            }
        
        return {"should_archive": False, "reason": "no_archival_criteria_met"}
    
    def promote_to_production(self, experiment_id: str, production_name: str = None) -> bool:
        """Manually promote an experiment to production."""
        
        exp_dir = self.experiments_dir / experiment_id
        if not exp_dir.exists():
            print(f"❌ Experiment {experiment_id} not found")
            return False
        
        # Load experiment metadata
        with open(exp_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        if metadata.get("status") != "completed":
            print(f"❌ Experiment {experiment_id} is not completed")
            return False
        
        # Create production name
        if not production_name:
            production_name = f"{metadata['experiment_name']}_production_v{datetime.now().strftime('%Y%m%d')}"
        
        production_path = self.production_dir / production_name
        
        # Move to production
        shutil.move(str(exp_dir), str(production_path))
        
        # Update metadata
        metadata["consolidation_status"] = "promoted_to_production"
        metadata["promoted_at"] = datetime.now().isoformat()
        metadata["production_name"] = production_name
        
        with open(production_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Update registry
        with open(self.models_registry, "r") as f:
            registry = json.load(f)
        
        registry["production_models"][production_name] = {
            "original_experiment": experiment_id,
            "promoted_at": datetime.now().isoformat(),
            "performance": metadata.get("final_metrics", {}),
            "architecture": metadata.get("config", {}).get("model", {}),
            "status": "production_ready"
        }
        
        registry["tracking_metadata"]["experiments_promoted_to_production"] += 1
        
        with open(self.models_registry, "w") as f:
            json.dump(registry, f, indent=2)
        
        print(f"🏆 Promoted {experiment_id} to production as {production_name}")
        return True
    
    def archive_experiment(self, experiment_id: str, archive_location: str = "archive_baseline") -> bool:
        """Manually archive an experiment."""
        
        exp_dir = self.experiments_dir / experiment_id
        if not exp_dir.exists():
            print(f"❌ Experiment {experiment_id} not found")
            return False
        
        # Choose archive directory
        if archive_location == "archive_debug":
            archive_dir = self.archive_debug_dir
        else:
            archive_dir = self.archive_baseline_dir
        
        archive_path = archive_dir / experiment_id
        
        # Move to archive
        shutil.move(str(exp_dir), str(archive_path))
        
        # Update metadata
        metadata_path = archive_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            metadata["consolidation_status"] = "archived"
            metadata["archived_at"] = datetime.now().isoformat()
            metadata["archive_location"] = archive_location
            
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        # Update registry
        with open(self.models_registry, "r") as f:
            registry = json.load(f)
        
        registry["archived_experiments"][archive_location][experiment_id] = {
            "archived_at": datetime.now().isoformat(),
            "original_location": f"experiments/{experiment_id}",
            "archive_reason": "manual_archive"
        }
        
        registry["tracking_metadata"]["experiments_auto_archived"] += 1
        
        with open(self.models_registry, "w") as f:
            json.dump(registry, f, indent=2)
        
        print(f"📦 Archived {experiment_id} to {archive_location}")
        return True
    
    def get_experiment_summary(self) -> Dict:
        """Get comprehensive experiment summary."""
        
        # Load registry with error handling
        try:
            with open(self.models_registry, "r") as f:
                registry = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If registry doesn't exist or is corrupted, return basic summary
            registry = {}
        
        # Count experiments in each location
        active_experiments = len([d for d in os.listdir(self.experiments_dir) 
                                if (self.experiments_dir / d).is_dir() and not d.startswith('archive')])
        
        archive_debug_count = len(os.listdir(self.archive_debug_dir)) if self.archive_debug_dir.exists() else 0
        archive_baseline_count = len(os.listdir(self.archive_baseline_dir)) if self.archive_baseline_dir.exists() else 0
        production_count = len(os.listdir(self.production_dir)) if self.production_dir.exists() else 0
        
        return {
            "active_experiments": active_experiments,
            "production_models": production_count,
            "archived_debug": archive_debug_count,
            "archived_baseline": archive_baseline_count,
            "total_experiments": active_experiments + archive_debug_count + archive_baseline_count + production_count,
            "registry_metadata": registry.get("tracking_metadata", {}),
            "last_updated": registry.get("last_updated")
        }
    
    def cleanup_old_debug_experiments(self, keep_latest: int = 1) -> int:
        """Clean up old debug experiments, keeping only the latest N."""
        
        debug_experiments = [d for d in os.listdir(self.experiments_dir) 
                            if any(keyword in d.lower() for keyword in ["debug", "test", "mock", "ultra-micro", "micro"])]
        
        if len(debug_experiments) <= keep_latest:
            return 0
        
        # Sort by creation time and archive older ones
        exp_info = []
        for exp_id in debug_experiments:
            exp_dir = self.experiments_dir / exp_id
            if exp_dir.exists():
                try:
                    with open(exp_dir / "metadata.json", "r") as f:
                        metadata = json.load(f)
                    created_at = datetime.fromisoformat(metadata["created_at"])
                    exp_info.append((exp_id, created_at))
                except:
                    continue
        
        # Sort by creation time (newest first)
        exp_info.sort(key=lambda x: x[1], reverse=True)
        
        # Archive older experiments
        archived_count = 0
        for exp_id, _ in exp_info[keep_latest:]:
            if self.archive_experiment(exp_id, "archive_debug"):
                archived_count += 1
        
        return archived_count
    
    # Include all methods from original tracker
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
        
        # Also save metrics to JSONL for easy analysis
        metrics_file = exp_dir / "metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metric_entry) + "\n")
    
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
    
    def compare_experiments(self, experiment_ids: List[str], 
                           metrics: List[str] = None) -> pd.DataFrame:
        """Compare experiments across all locations."""
        # ... (enhanced version that looks in archives too)
        pass
    
    # Helper methods
    def _log_experiment(self, metadata: Dict):
        """Log experiment to JSONL file."""
        log_entry = {
            "experiment_id": metadata["experiment_id"],
            "timestamp": metadata["created_at"],
            "experiment_name": metadata["experiment_name"],
            "description": metadata["description"],
            "config_hash": metadata["config_hash"],
            "tags": metadata["tags"],
            "status": metadata["status"],
            "experiment_category": metadata.get("experiment_category"),
            "tracking_version": metadata.get("tracking_version")
        }
        
        with open(self.experiments_log, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def _update_active_experiments(self, experiment_id: str, metadata: Dict):
        """Update active experiments in registry."""
        with open(self.models_registry, "r") as f:
            registry = json.load(f)
        
        # Ensure active_experiments key exists
        if "active_experiments" not in registry:
            registry["active_experiments"] = {}
        
        # Ensure tracking_metadata key exists
        if "tracking_metadata" not in registry:
            registry["tracking_metadata"] = {
                "total_experiments_run": 0,
                "experiments_promoted_to_production": 0,
                "experiments_auto_archived": 0
            }
        
        registry["active_experiments"][experiment_id] = {
            "experiment_name": metadata["experiment_name"],
            "created_at": metadata["created_at"],
            "experiment_category": metadata.get("experiment_category"),
            "status": metadata["status"]
        }
        
        registry["tracking_metadata"]["total_experiments_run"] += 1
        registry["last_updated"] = datetime.now().isoformat()
        
        with open(self.models_registry, "w") as f:
            json.dump(registry, f, indent=2)
    
    def _update_consolidated_registry(self, experiment_id: str, metadata: Dict, 
                                    promotion_result: Dict, archival_result: Dict):
        """Update consolidated registry after experiment completion."""
        with open(self.models_registry, "r") as f:
            registry = json.load(f)
        
        # Ensure active_experiments key exists
        if "active_experiments" not in registry:
            registry["active_experiments"] = {}
        
        # Update active experiments
        if experiment_id in registry["active_experiments"]:
            registry["active_experiments"][experiment_id].update({
                "status": metadata["status"],
                "completed_at": metadata.get("completed_at"),
                "final_metrics": metadata.get("final_metrics", {}),
                "promotion_candidate": promotion_result.get("promotion_candidate", False),
                "archival_pending": archival_result.get("should_archive", False)
            })
        else:
            # If experiment not in registry, add it
            registry["active_experiments"][experiment_id] = {
                "experiment_name": metadata.get("experiment_name", ""),
                "status": metadata["status"],
                "completed_at": metadata.get("completed_at"),
                "final_metrics": metadata.get("final_metrics", {}),
                "promotion_candidate": promotion_result.get("promotion_candidate", False),
                "archival_pending": archival_result.get("should_archive", False)
            }
        
        registry["last_updated"] = datetime.now().isoformat()
        
        with open(self.models_registry, "w") as f:
            json.dump(registry, f, indent=2)
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None
    
    def _hash_config(self, config: Dict) -> str:
        """Create hash of configuration for tracking."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

# Create consolidated tracker instance
consolidated_tracker = ConsolidatedExperimentTracker()
