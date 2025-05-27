# Mental Health Classifier - Model Consolidation Report

Generated: 2025-05-25

## Current State Analysis

### Directory Structure Overview
- **experiments/**: 23 experiment runs (various sizes and configurations)
- **saved_models/**: Legacy model files (simple_trained_model.pth, vocab.pkl)
- **models/**: Current production model (best_model.pt, model_info.json, vocab.pkl)
- **model_snapshots/**: Single baseline snapshot
- **models_registry.json**: Registry with minimal tracking

### Experiment Categories Identified

#### 1. Production-Ready Models
- **small_data_10k_20250525_130522**: Best balanced performance
  - Accuracy: 58.2%, F1-macro: 53.0%
  - Architecture: 3-layer, 256-dim, 10K vocab
  - Status: Completed successfully
  
- **medium_model_v2_20250525_131811**: Larger capacity model
  - Accuracy: 56.3%, F1-macro: 45.3%
  - Architecture: 6-layer, 512-dim, 15K vocab
  - Performance issues with depression class (0% precision/recall)

#### 2. Debug/Test Models (Safe to Remove)
- **Ultra-micro models** (11 runs): Minimal architectures for debugging
  - baseline_ultra_micro_* series
  - Very small vocab (100), single layer
  - High accuracy (99.5%) due to overfitting on tiny dataset
  
- **Micro models** (3 runs): Small debug models
  - baseline_micro_* series
  - Used for debugging bus errors and system issues

#### 3. Baseline Models
- **baseline_small_* series** (3 runs): Reproducibility tests
- **baseline_v1_20250525_064536**: Original baseline snapshot

#### 4. Test/Mock Models
- **mock_test_20250525_112344**: Testing infrastructure
- **safe_test_*, minimal_test_***: Development testing

## Consolidation Recommendations

### Keep (Production Models)
1. **small_data_10k_20250525_130522** → Rename to `production_baseline_v1`
2. **medium_model_v2_20250525_131811** → Rename to `production_medium_v1` 
3. **baseline_v1_20250525_064536** → Keep as historical baseline

### Archive (Reference Models)
Move to `experiments/archived/`:
- One representative from each series (baseline_small, baseline_micro)
- Keep metadata but remove model files to save space

### Remove (Debug/Test Models)
Delete entirely:
- All ultra_micro experiments (11 runs)
- Duplicate micro experiments (keep 1)
- Mock/test experiments
- Failed/incomplete runs

### Storage Optimization
- Estimated space savings: ~2-3 GB
- Keep only essential model checkpoints
- Compress archived experiments

## Implementation Plan

### Phase 1: Create Production Model Archive
1. Create `models/production/` directory
2. Move best performing models with standardized naming
3. Update registry with production model status

### Phase 2: Archive Historical Models
1. Create `experiments/archived/` directory
2. Move representative samples from each experiment type
3. Compress model files for storage efficiency

### Phase 3: Clean Debug Models
1. Remove all ultra-micro and debug experiments
2. Clean up duplicate micro experiments
3. Remove test/mock experiments

### Phase 4: Update Registry
1. Update models_registry.json with production model metadata
2. Add performance benchmarks and recommendations
3. Document model selection criteria

## Performance Summary

| Model | Architecture | Accuracy | F1-Macro | Best For |
|-------|-------------|----------|----------|----------|
| small_data_10k | 3L-256D-10K | 58.2% | 53.0% | Balanced performance |
| medium_model_v2 | 6L-512D-15K | 56.3% | 45.3% | High capacity (needs tuning) |
| baseline_v1 | 3L-256D | - | - | Historical reference |

## Next Steps
1. Execute consolidation plan
2. Retrain medium model with better hyperparameters
3. Implement model versioning system
4. Set up automated cleanup for debug runs
