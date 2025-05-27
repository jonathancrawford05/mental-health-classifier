# Model Reproducibility and Experiment Tracking Guide

## 🎯 Overview

This system provides comprehensive experiment tracking and model versioning for systematic architecture exploration. Every experiment is preserved with full reproducibility.

## 📁 New Directory Structure

```
mental-health-classifier/
├── experiments/                    # All experiment results
│   ├── baseline_v1_20241225/      # Timestamped experiments
│   ├── medium_model_20241225/     
│   └── umls_enhanced_20241225/    
├── model_snapshots/               # Model preservation snapshots
├── scripts/                       # Utility scripts
│   ├── preserve_current_model.py  # Snapshot current state
│   └── analyze_experiments.py     # Compare experiments
├── config/
│   ├── config.yaml               # Base configuration
│   └── experiment_configs.yaml   # Architecture templates
├── experiments_log.jsonl         # Experiment tracking log
├── models_registry.json          # Best model registry
└── train_with_tracking.py        # Enhanced training script
```

## 🚀 Quick Start

### 1. Preserve Your Current Model (IMPORTANT - Do This First!)

```bash
# This creates a snapshot of your current model before any changes
python scripts/preserve_current_model.py
```

This creates a timestamped snapshot in `model_snapshots/` with:
- Complete source code
- Configuration files  
- Any trained models
- Git commit information
- Metadata for future reference

### 2. Run Systematic Experiments

#### Train with Different Architectures:

```bash
# Baseline (current small model)
python train_with_tracking.py --experiment-config baseline_small --create-sample-data --sample-size 1000

# Medium model (scaled up)
python train_with_tracking.py --experiment-config medium_model --create-sample-data --sample-size 2000

# Large model (full scale)  
python train_with_tracking.py --experiment-config large_model --create-sample-data --sample-size 3000

# UMLS enhanced model
python train_with_tracking.py --experiment-config umls_enhanced --create-sample-data --sample-size 2000
```

#### Custom Experiments:

```bash
# Custom experiment with specific description
python train_with_tracking.py \
    --experiment-name "custom_dropout_test" \
    --description "Testing higher dropout for regularization" \
    --tags "dropout" "regularization" "test" \
    --create-sample-data
```

### 3. Compare and Analyze Results

```bash
# Compare all experiments
python scripts/analyze_experiments.py --compare

# Plot architecture analysis
python scripts/analyze_experiments.py --plot-architecture --metric final_accuracy

# Clinical performance analysis (suicide detection focus)
python scripts/analyze_experiments.py --clinical-analysis

# Generate comprehensive HTML report
python scripts/analyze_experiments.py --generate-report
```

## 📊 Available Experiment Configurations

The system includes pre-configured architectures in `config/experiment_configs.yaml`:

1. **baseline_small** - Current architecture (256 dim, 3 layers)
2. **medium_model** - Scaled up (512 dim, 6 layers)  
3. **large_model** - Full scale (768 dim, 12 layers)
4. **umls_enhanced** - Clinical vocabulary integration
5. **curriculum_model** - Progressive training approach
6. **multitask_model** - Multi-task learning setup
7. **interpretable_model** - Attention analysis optimized

## 🔍 Experiment Tracking Features

### Automatic Tracking:
- **Architecture Parameters**: Embedding dim, layers, heads, dropout
- **Training Configuration**: Learning rate, batch size, epochs  
- **Performance Metrics**: Accuracy, F1, precision, recall (per class)
- **Training Time**: Duration and computational efficiency
- **Model Size**: Parameter count and memory usage
- **Git Information**: Commit hash, branch, dirty status

### Clinical Focus:
- **Suicide Risk Recall**: Prioritized for clinical safety
- **Class-Specific Metrics**: Individual performance per disorder
- **Clinical Vocabulary Impact**: UMLS integration effectiveness
- **Interpretability Scores**: Attention analysis results

## 📈 Best Model Registry

The system automatically tracks:
- **Highest Overall Accuracy**
- **Best F1 Macro Score**  
- **Best Suicide Risk Recall** (most clinically important)
- **Most Stable Model** (consistent across runs)

Access via:
```python
from src.utils.experiment_tracker import tracker
best_models = tracker.get_best_models()
```

## 🔄 Reproducibility Features

### Full Snapshot System:
- Complete code state preservation
- Configuration versioning
- Environment capture
- Git commit tracking
- Dependency versioning

### Deterministic Training:
- Fixed random seeds
- Deterministic algorithms where possible
- Environment variable settings
- Hardware configuration logging

## 📋 Analysis Outputs

### Comparison Table:
- Side-by-side performance comparison
- Architecture parameter analysis
- Training efficiency metrics
- Clinical performance focus

### Visualization:
- Architecture vs Performance plots
- Training curve comparisons  
- Clinical analysis charts
- Parameter efficiency analysis

### HTML Report:
- Executive summary
- Best model recommendations
- Architecture insights
- Clinical deployment guidance

## 🎛️ Advanced Usage

### Programmatic Access:

```python
from src.utils.experiment_tracker import tracker

# Start custom experiment
exp_id = tracker.start_experiment(
    experiment_name="custom_test",
    description="Testing new architecture",
    config=config,
    tags=["custom", "test"]
)

# Log metrics during training
tracker.log_metrics(exp_id, {"accuracy": 0.85, "loss": 0.3}, epoch=5)

# Save model checkpoint
tracker.save_model(exp_id, model, "checkpoint_epoch_10")

# Finish experiment
tracker.finish_experiment(exp_id, final_metrics={"accuracy": 0.90})
```

### Batch Experiment Comparison:

```python
from scripts.analyze_experiments import ModelAnalyzer

analyzer = ModelAnalyzer()
comparison_df = analyzer.create_performance_comparison()

# Filter experiments
best_models = comparison_df[comparison_df['final_accuracy'] > 0.85]
print(best_models[['experiment_name', 'final_accuracy', 'parameters_millions']])
```

## 🚨 Important Notes

### Before Making Changes:
1. **Always run** `python scripts/preserve_current_model.py` first
2. **Commit your code** to git before major experiments
3. **Document experiment hypotheses** in descriptions
4. **Use meaningful tags** for easy filtering

### Clinical Considerations:
- **Suicide recall is prioritized** over overall accuracy
- **False negatives are more dangerous** than false positives
- **Model interpretability** is crucial for clinical acceptance  
- **Ensemble methods** recommended for production

### Resource Management:
- Larger models require more memory and time
- Use appropriate batch sizes for your hardware
- Monitor GPU/CPU usage during training
- Clean up old experiments periodically

## 📞 Support

If you encounter issues:
1. Check experiment logs in `experiments/{experiment_id}/logs/`
2. Review configuration in `experiments/{experiment_id}/config/`
3. Compare with working baseline experiments
4. Use the analysis tools to debug performance issues

## 🎯 Next Steps

1. **Preserve current model** (critical first step)
2. **Run baseline experiment** to establish performance floor
3. **Scale up systematically** using provided configurations
4. **Integrate UMLS vocabulary** for clinical enhancement
5. **Analyze results** to identify best architectures
6. **Deploy best model** with appropriate safeguards
