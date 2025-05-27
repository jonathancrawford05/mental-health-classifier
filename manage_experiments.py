#!/usr/bin/env python3
"""
Consolidated Experiment Management Script
Easy management of experiments, archiving, and production promotion
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.consolidated_experiment_tracker import consolidated_tracker


def list_experiments():
    """List all experiments across different locations."""
    print("🔍 Experiment Inventory")
    print("=" * 50)
    
    summary = consolidated_tracker.get_experiment_summary()
    
    print(f"📊 Summary:")
    print(f"  Active experiments: {summary['active_experiments']}")
    print(f"  Production models: {summary['production_models']}")
    print(f"  Archived (debug): {summary['archived_debug']}")
    print(f"  Archived (baseline): {summary['archived_baseline']}")
    print(f"  Total experiments: {summary['total_experiments']}")
    
    # List active experiments
    experiments_dir = Path("experiments")
    if summary['active_experiments'] > 0:
        print(f"\n📁 Active Experiments:")
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith('archive'):
                metadata_path = exp_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        status = metadata.get('status', 'unknown')
                        created = metadata.get('created_at', '')[:10]  # Just date
                        description = metadata.get('description', '')[:50]
                        
                        print(f"  - {exp_dir.name} ({status}) - {created} - {description}")
                    except Exception as e:
                        print(f"  - {exp_dir.name} (error reading metadata)")
    
    # List production models
    production_dir = Path("models/production")
    if production_dir.exists() and summary['production_models'] > 0:
        print(f"\n🏆 Production Models:")
        for prod_dir in production_dir.iterdir():
            if prod_dir.is_dir():
                metadata_path = prod_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        final_metrics = metadata.get('final_metrics', {})
                        accuracy = final_metrics.get('accuracy', 0)
                        f1_macro = final_metrics.get('f1_macro', 0)
                        
                        print(f"  - {prod_dir.name} (Acc: {accuracy:.3f}, F1: {f1_macro:.3f})")
                    except:
                        print(f"  - {prod_dir.name} (metadata unavailable)")


def show_experiment_details(experiment_id: str):
    """Show detailed information about a specific experiment."""
    
    # Check different locations
    locations = [
        (Path("experiments") / experiment_id, "active"),
        (Path("models/production") / experiment_id, "production"),
        (Path("experiments/archive_debug") / experiment_id, "archived_debug"),
        (Path("experiments/archive_baseline") / experiment_id, "archived_baseline"),
    ]
    
    exp_path = None
    location = None
    
    for path, loc in locations:
        if path.exists():
            exp_path = path
            location = loc
            break
    
    if not exp_path:
        print(f"❌ Experiment '{experiment_id}' not found in any location")
        return
    
    print(f"🔍 Experiment Details: {experiment_id}")
    print(f"📍 Location: {location}")
    print("=" * 60)
    
    # Load and display metadata
    metadata_path = exp_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"📝 Name: {metadata.get('experiment_name', 'N/A')}")
        print(f"📄 Description: {metadata.get('description', 'N/A')}")
        print(f"🏷️  Tags: {', '.join(metadata.get('tags', []))}")
        print(f"📅 Created: {metadata.get('created_at', 'N/A')}")
        print(f"✅ Status: {metadata.get('status', 'N/A')}")
        
        if metadata.get('completed_at'):
            print(f"🏁 Completed: {metadata['completed_at']}")
            duration = metadata.get('total_duration_seconds', 0)
            print(f"⏱️  Duration: {duration/60:.1f} minutes")
        
        # Show final metrics
        final_metrics = metadata.get('final_metrics', {})
        if final_metrics:
            print(f"\n📊 Final Metrics:")
            for metric, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
        
        # Show model architecture
        model_config = metadata.get('config', {}).get('model', {})
        if model_config:
            print(f"\n🏗️  Model Architecture:")
            print(f"  Layers: {model_config.get('n_layer', 'N/A')}")
            print(f"  Embedding dim: {model_config.get('n_embd', 'N/A')}")
            print(f"  Attention heads: {model_config.get('num_heads', 'N/A')}")
            print(f"  Vocab size: {model_config.get('vocab_size', 'N/A')}")
    
    # Show available files
    print(f"\n📁 Available Files:")
    for item in exp_path.iterdir():
        if item.is_dir():
            file_count = len(list(item.iterdir()))
            print(f"  📂 {item.name}/ ({file_count} files)")
        else:
            print(f"  📄 {item.name}")


def promote_experiment(experiment_id: str, production_name: str = None):
    """Promote an experiment to production."""
    print(f"🏆 Promoting experiment {experiment_id} to production...")
    
    success = consolidated_tracker.promote_to_production(experiment_id, production_name)
    
    if success:
        print(f"✅ Successfully promoted to production!")
        summary = consolidated_tracker.get_experiment_summary()
        print(f"📊 Production models: {summary['production_models']}")
    else:
        print(f"❌ Failed to promote experiment")


def archive_experiment(experiment_id: str, archive_type: str = "archive_baseline"):
    """Archive an experiment."""
    print(f"📦 Archiving experiment {experiment_id} to {archive_type}...")
    
    success = consolidated_tracker.archive_experiment(experiment_id, archive_type)
    
    if success:
        print(f"✅ Successfully archived experiment!")
        summary = consolidated_tracker.get_experiment_summary()
        print(f"📊 Archived experiments: {summary['archived_debug'] + summary['archived_baseline']}")
    else:
        print(f"❌ Failed to archive experiment")


def cleanup_debug_experiments(keep_latest: int = 2):
    """Clean up old debug experiments."""
    print(f"🧹 Cleaning up debug experiments (keeping latest {keep_latest})...")
    
    archived_count = consolidated_tracker.cleanup_old_debug_experiments(keep_latest)
    
    if archived_count > 0:
        print(f"✅ Archived {archived_count} old debug experiments")
    else:
        print(f"ℹ️  No debug experiments to archive")


def show_production_candidates():
    """Show experiments that could be promoted to production."""
    print("🏆 Production Promotion Candidates")
    print("=" * 40)
    
    experiments_dir = Path("experiments")
    candidates = []
    
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir() and not exp_dir.name.startswith('archive'):
            metadata_path = exp_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    if metadata.get('status') == 'completed':
                        final_metrics = metadata.get('final_metrics', {})
                        accuracy = final_metrics.get('accuracy', 0)
                        f1_macro = final_metrics.get('f1_macro', 0)
                        
                        # Check promotion thresholds
                        if accuracy >= 0.55 and f1_macro >= 0.45:
                            candidates.append({
                                'id': exp_dir.name,
                                'name': metadata.get('experiment_name', ''),
                                'accuracy': accuracy,
                                'f1_macro': f1_macro,
                                'created': metadata.get('created_at', '')[:10]
                            })
                except:
                    continue
    
    if candidates:
        print(f"Found {len(candidates)} candidates:")
        for candidate in sorted(candidates, key=lambda x: x['f1_macro'], reverse=True):
            print(f"  ✅ {candidate['id']}")
            print(f"     Accuracy: {candidate['accuracy']:.3f}, F1: {candidate['f1_macro']:.3f}")
            print(f"     Created: {candidate['created']}")
            print(f"     Promote: python manage_experiments.py promote {candidate['id']}")
            print()
    else:
        print("❌ No experiments meet production thresholds (Acc ≥ 0.55, F1 ≥ 0.45)")


def main():
    parser = argparse.ArgumentParser(description="Manage consolidated experiments")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    subparsers.add_parser('list', help='List all experiments')
    
    # Details command
    details_parser = subparsers.add_parser('details', help='Show experiment details')
    details_parser.add_argument('experiment_id', help='Experiment ID to show details for')
    
    # Promote command
    promote_parser = subparsers.add_parser('promote', help='Promote experiment to production')
    promote_parser.add_argument('experiment_id', help='Experiment ID to promote')
    promote_parser.add_argument('--name', help='Production model name (optional)')
    
    # Archive command
    archive_parser = subparsers.add_parser('archive', help='Archive an experiment')
    archive_parser.add_argument('experiment_id', help='Experiment ID to archive')
    archive_parser.add_argument('--type', choices=['archive_debug', 'archive_baseline'], 
                               default='archive_baseline', help='Archive type')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old debug experiments')
    cleanup_parser.add_argument('--keep', type=int, default=2, help='Number of latest debug experiments to keep')
    
    # Candidates command
    subparsers.add_parser('candidates', help='Show production promotion candidates')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'list':
            list_experiments()
        elif args.command == 'details':
            show_experiment_details(args.experiment_id)
        elif args.command == 'promote':
            promote_experiment(args.experiment_id, args.name)
        elif args.command == 'archive':
            archive_experiment(args.experiment_id, args.type)
        elif args.command == 'cleanup':
            cleanup_debug_experiments(args.keep)
        elif args.command == 'candidates':
            show_production_candidates()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
