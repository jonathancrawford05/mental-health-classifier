# Model Consolidation - Git Tracking Strategy

## Current Git Status
- Repository: https://github.com/jonathancrawford05/mental-health-classifier.git
- Branch: experiment-current-model-eval (with backup-before-cleanup branch)
- `.gitignore` excludes: models/, *.pt, *.pkl, data/, results/, logs/

## Git-Safe Consolidation Approach

### Phase 1: Document Current State (Track in Git)
1. **Create consolidation documentation** ✅
   - `model_consolidation_report.md` 
   - `models_registry_consolidated.json`
   - Track experiment metadata and performance metrics

2. **Update .gitignore for better tracking**
   - Allow tracking of metadata files in models/
   - Track consolidation scripts

### Phase 2: Archive Strategy (Git + File System)
Instead of deleting, move experiments to archive folders:
- `experiments/archived/` - Keep important experiments
- `experiments/debug/` - Move debug experiments
- Document moves in git-tracked files

### Phase 3: Metadata Tracking (Git Tracked)
- Track experiment metadata (JSON files)
- Track model configurations
- Track performance summaries
- Ignore large binary model files

## Benefits of This Approach
1. **Version Control**: All consolidation decisions documented in git
2. **Reversible**: Can restore experiments if needed
3. **Collaborative**: Team can see consolidation rationale
4. **Audit Trail**: Clear history of model evolution

## Files to Track in Git
- ✅ Documentation: `*.md` files
- ✅ Metadata: `*.json` files (experiment configs, performance)
- ✅ Scripts: consolidation and cleanup scripts
- ✅ Registry: updated model registry
- ❌ Binaries: model weights, vocabularies (too large)

## Recommended Git Workflow
```bash
# 1. Create consolidation branch
git checkout -b model-consolidation

# 2. Add documentation files
git add model_consolidation_report.md
git add models_registry_consolidated.json
git add consolidate_models.py

# 3. Update .gitignore if needed
git add .gitignore

# 4. Commit consolidation documentation
git commit -m "feat: Model consolidation documentation and registry

- Add comprehensive model consolidation report
- Create production model registry with performance metrics  
- Document consolidation strategy and recommendations
- Add safe cleanup scripts for debug experiments

Models moved to production/:
- baseline_v1: 58.2% accuracy, balanced performance
- medium_v1: 56.3% accuracy, needs depression class tuning"

# 5. Push to create PR
git push -u origin model-consolidation
```

## Next Steps
1. Review current git status
2. Create consolidation branch
3. Track documentation in git
4. Move experiments to archive folders
5. Update registry with new locations
