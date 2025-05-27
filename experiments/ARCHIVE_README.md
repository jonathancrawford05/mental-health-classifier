# Archived Experiments

This directory contains archived experiments from the model consolidation process.

## Archive Categories

### `archive_debug/`
Debug and test experiments used for development and troubleshooting:
- Ultra-micro models (minimal architectures for debugging)
- Mock test experiments  
- Safe test experiments
- Development prototypes

**Purpose**: These experiments were used to debug system issues, test infrastructure, and validate training pipelines. They're archived for reference but not recommended for production use.

### `archive_baseline/`
Baseline and variant experiments:
- Duplicate baseline runs
- Alternative architectures
- Data size experiments  
- Production variants

**Purpose**: These experiments represent different approaches and configurations tested during development. They're kept for comparative analysis and reproducibility.

### `archived/` (Reference Models)
Key reference implementations:
- `baseline_small_reference/` - Reproducibility reference
- Other important baseline implementations

**Purpose**: These are clean reference implementations kept for comparison and reproducibility studies.

## Archive Structure
```
experiments/
├── archive_debug/          # Debug and test experiments
├── archive_baseline/       # Baseline variants and alternatives  
├── archived/              # Key reference implementations
└── debug_archive/         # Minimal debug references
```

## Accessing Archived Experiments
All archived experiments retain their full structure:
- `metadata.json` - Experiment configuration and results
- `metrics.jsonl` - Training metrics timeline
- `models/` - Model checkpoints (if any)
- `results/` - Evaluation results
- `logs/` - Training logs

## Consolidation Documentation
See the following files in the project root for full consolidation details:
- `model_consolidation_report.md` - Comprehensive analysis
- `models_registry_consolidated.json` - Model registry with performance metrics
- `experiment_consolidation_log.json` - Move tracking and metadata
- `git_consolidation_strategy.md` - Git workflow documentation

## Production Models
Production-ready models have been moved to `models/production/`:
- `baseline_v1/` - Best balanced performance (58.2% accuracy)
- `medium_v1/` - High capacity model (needs tuning)

---
*Archive created: 2025-05-25*  
*Consolidation version: v1.0*
