# New Experiment System - Setup & Usage Guide

## 🎯 **What's New After Consolidation**

Your experiment tracking system has been upgraded to work seamlessly with the consolidated model structure. Here's what's changed and how to use it:

### **📁 New Directory Structure**
```
mental-health-classifier/
├── experiments/
│   ├── archive_debug/          # Auto-archived debug experiments
│   ├── archive_baseline/       # Archived baseline experiments  
│   ├── archived/               # Reference experiments
│   └── [active_experiments]/   # Current running/completed experiments
├── models/
│   └── production/             # Production-ready models
│       ├── baseline_v1/        # Your best model (58.2% accuracy)
│       └── medium_v1/          # High capacity model (needs tuning)
└── src/utils/
    └── consolidated_experiment_tracker.py  # New enhanced tracker
```

### **🔄 Enhanced Experiment Tracking**

#### **New Features:**
- **Auto-categorization**: Experiments are automatically classified as "debug" or "baseline"
- **Auto-archiving**: Debug experiments are automatically moved to archive folders
- **Production promotion**: Automatic detection of models ready for production
- **Consolidated registry**: Single source of truth for all model metadata

## 🚀 **Usage Instructions**

### **1. Run New Experiments**

Use the new training script with enhanced tracking:

```bash
# Basic training with consolidated tracking
python train_with_consolidated_tracking.py \
    --experiment-name "my_new_model" \
    --description "Testing improved architecture" \
    --tags baseline production-candidate

# With experiment summary first
python train_with_consolidated_tracking.py \
    --experiment-summary \
    --experiment-name "debug_test" \
    --tags debug micro \
    --auto-archive-debug

# Check for production promotion after training
python train_with_consolidated_tracking.py \
    --experiment-name "candidate_model" \
    --check-production-promotion
```

### **2. Manage Experiments**

Use the new management script for easy experiment handling:

```bash
# List all experiments across all locations
python manage_experiments.py list

# Show detailed info about a specific experiment
python manage_experiments.py details experiment_name_20250525_123456

# Show which experiments are ready for production
python manage_experiments.py candidates

# Promote an experiment to production
python manage_experiments.py promote experiment_name_20250525_123456

# Archive an old experiment
python manage_experiments.py archive old_experiment_20250525_111111 --type archive_debug

# Clean up old debug experiments (keep only latest 2)
python manage_experiments.py cleanup --keep 2
```

### **3. Production Model Management**

```bash
# Promote experiment to production manually
python -c "
from src.utils.consolidated_experiment_tracker import consolidated_tracker
consolidated_tracker.promote_to_production('experiment_id', 'production_name_v2')
"

# Check production model status
python manage_experiments.py list
```

## 🎛️ **Auto Features**

### **Automatic Experiment Classification**
- **Debug experiments**: Names/tags containing "debug", "test", "mock", "micro", "minimal"
- **Baseline experiments**: Everything else, including "baseline", "production", "medium"

### **Automatic Archiving Rules**
- **Debug experiments**: Auto-archived when >3 debug experiments exist
- **Failed experiments**: Auto-archived after 7 days
- **Configurable**: Edit rules in `models_registry_consolidated.json`

### **Production Promotion Criteria**
- **Accuracy**: ≥ 55%
- **F1-Macro**: ≥ 45%  
- **Manual approval**: Required by default (configurable)

## 📊 **Monitoring & Status**

### **Check System Status**
```bash
# Quick status overview
python manage_experiments.py list

# Detailed experiment info
python manage_experiments.py details your_experiment_id

# Find production candidates
python manage_experiments.py candidates
```

### **Registry Files**
- `models_registry_consolidated.json` - Master registry with all model metadata
- `experiment_consolidation_log.json` - Tracks moves and archiving decisions
- `experiments_log.jsonl` - Chronological log of all experiments

## 🔧 **Configuration**

### **Auto-Archiving Rules** (in `models_registry_consolidated.json`)
```json
"auto_archival_rules": {
  "debug_experiments": {
    "enabled": true,
    "criteria": ["debug", "test", "mock", "ultra-micro", "micro"],
    "min_experiments_before_archive": 3,
    "archive_location": "archive_debug"
  },
  "production_promotion": {
    "enabled": true,
    "min_accuracy_threshold": 0.55,
    "min_f1_macro_threshold": 0.45,
    "require_manual_approval": true
  }
}
```

## 📝 **Example Workflows**

### **1. Research Experiment**
```bash
# Run experiment with detailed tracking
python train_with_consolidated_tracking.py \
    --experiment-name "research_architecture_v3" \
    --description "Testing attention mechanism improvements" \
    --tags baseline research attention \
    --experiment-summary \
    --check-production-promotion

# If it performs well, promote it
python manage_experiments.py promote research_architecture_v3_20250525_140000
```

### **2. Debug/Test Experiment**  
```bash
# Run debug experiment (auto-classified as debug)
python train_with_consolidated_tracking.py \
    --experiment-name "debug_data_loader" \
    --description "Testing new data loading pipeline" \
    --tags debug test \
    --auto-archive-debug

# System automatically archives older debug experiments
```

### **3. Regular Maintenance**
```bash
# Weekly cleanup
python manage_experiments.py cleanup --keep 2

# Check for new production candidates
python manage_experiments.py candidates

# Review experiment status
python manage_experiments.py list
```

## ✅ **Migration from Old System**

Your old experiments are preserved in the archive structure:

1. **Production models**: Moved to `models/production/`
2. **Debug experiments**: Moved to `experiments/archive_debug/`
3. **Baseline experiments**: Moved to `experiments/archive_baseline/`
4. **All metadata preserved** for future reference
5. **Git history maintained** for all consolidation decisions

## 🎉 **Benefits of the New System**

### **For Development**
- **Automatic organization**: No more manual experiment cleanup
- **Clear production path**: Automatic identification of deployment-ready models
- **Better experiment discovery**: Easy search across all locations
- **Consistent metadata**: Standardized tracking across all experiments

### **For Production**
- **Clear model registry**: Single source of truth for production models
- **Performance benchmarks**: Automatic comparison and ranking
- **Audit trail**: Full history of model promotion decisions
- **Version control**: Git-tracked consolidation decisions

### **For Collaboration**
- **Team visibility**: Clear experiment status for all team members
- **Reproducibility**: Complete experiment metadata and configurations
- **Documentation**: Automatic generation of experiment reports
- **Decision tracking**: Rationale for all model decisions preserved

## 🚨 **Important Notes**

### **Breaking Changes**
- **New tracker required**: Use `train_with_consolidated_tracking.py` instead of old training script
- **Import changes**: Import from `consolidated_experiment_tracker` instead of `experiment_tracker`
- **Management workflow**: Use `manage_experiments.py` for experiment operations

### **Backward Compatibility**
- **Old experiments accessible**: All archived experiments retain full functionality
- **Metadata preserved**: All performance metrics and configurations available
- **Models loadable**: All model checkpoints remain accessible

## 🎯 **Next Steps for You**

### **Immediate (Today)**
1. **Test the new system**:
   ```bash
   python manage_experiments.py list
   python manage_experiments.py candidates
   ```

2. **Run a test experiment**:
   ```bash
   python train_with_consolidated_tracking.py \
       --experiment-name "test_new_system" \
       --description "Testing consolidated tracking" \
       --tags test \
       --experiment-summary
   ```

### **This Week**
3. **Review production models**:
   - Deploy `baseline_v1` if ready for production use
   - Retrain `medium_v1` with better hyperparameters for depression class

4. **Configure auto-archiving** rules to match your preferences

5. **Set up automated cleanup** in your workflow

### **Ongoing**
6. **Use new workflows** for all future experiments
7. **Monitor experiment registry** for production candidates
8. **Regular cleanup** of debug experiments

## 📞 **Support & Troubleshooting**

### **Common Issues**
- **Import errors**: Ensure `src/` is in Python path
- **Missing experiments**: Check archive directories with `manage_experiments.py list`
- **Permission errors**: Ensure write access to experiments/ and models/ directories

### **Debug Commands**
```bash
# Check system status
python -c "from src.utils.consolidated_experiment_tracker import consolidated_tracker; print(consolidated_tracker.get_experiment_summary())"

# Verify tracker initialization
python -c "from src.utils.consolidated_experiment_tracker import consolidated_tracker; print('Tracker initialized successfully')"

# Check registry file
cat models_registry_consolidated.json | jq '.tracking_metadata'
```

### **Recovery Procedures**
If something goes wrong:
1. All original experiments are preserved in archive folders
2. Git history contains all consolidation decisions
3. Registry files can be regenerated from experiment metadata
4. Production models are safely stored in `models/production/`

---

## 🎊 **You're All Set!**

Your experiment tracking system is now:
- ✅ **Consolidated** and organized
- ✅ **Git-tracked** with full audit trail
- ✅ **Production-ready** with clear deployment path
- ✅ **Future-proof** with automatic management

**Ready to run your next experiment?**
```bash
python train_with_consolidated_tracking.py --experiment-summary
```
