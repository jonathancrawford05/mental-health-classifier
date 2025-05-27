#!/bin/bash
# Git Consolidation Workflow
# Run this script to properly track the model consolidation in git

set -e  # Exit on any error

echo "🔄 Mental Health Classifier - Git Consolidation Workflow"
echo "========================================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d ".git" ]; then
    echo "❌ Error: Run this script from the mental-health-classifier root directory"
    exit 1
fi

# Check current git status
echo "📍 Current git status:"
git status --short

echo ""
echo "🌿 Current branch: $(git branch --show-current)"

# Suggest creating a consolidation branch
echo ""
echo "📝 Recommended workflow:"
echo "1. Create consolidation branch (if not already on one)"
echo "2. Add consolidation documentation files"  
echo "3. Commit consolidation changes"
echo "4. Archive experiments (optional)"

echo ""
read -p "Create 'model-consolidation' branch? (y/N): " create_branch

if [[ $create_branch =~ ^[Yy]$ ]]; then
    echo "🌿 Creating and switching to model-consolidation branch..."
    git checkout -b model-consolidation 2>/dev/null || git checkout model-consolidation
    echo "✅ On branch: $(git branch --show-current)"
fi

echo ""
echo "📁 Adding consolidation documentation files..."

# Add the consolidation files we created
files_to_add=(
    "model_consolidation_report.md"
    "models_registry_consolidated.json"
    "experiment_consolidation_log.json"
    "git_consolidation_strategy.md"
    "archive_experiments.py"
    "consolidate_models.py"
    "experiments/ARCHIVE_README.md"
)

added_files=()
for file in "${files_to_add[@]}"; do
    if [ -f "$file" ]; then
        git add "$file"
        added_files+=("$file")
        echo "  ✅ Added: $file"
    else
        echo "  ⚠️  Not found: $file"
    fi
done

echo ""
echo "📋 Files staged for commit:"
git diff --name-only --cached

echo ""
echo "💬 Commit message preview:"
echo "---"
cat << 'EOF'
feat: Model consolidation documentation and strategy

Add comprehensive model consolidation framework:

- model_consolidation_report.md: Analysis of 23 experiments with performance metrics
- models_registry_consolidated.json: Production model registry  
- experiment_consolidation_log.json: Detailed move tracking and metadata
- git_consolidation_strategy.md: Git workflow for consolidation
- archive_experiments.py: Safe archiving script (no deletion)
- experiments/ARCHIVE_README.md: Archive directory documentation

Production models identified:
- baseline_v1: 58.2% accuracy, balanced F1 scores (RECOMMENDED)
- medium_v1: 56.3% accuracy, needs depression class tuning

Archive strategy: Move experiments to archive directories instead of deletion
to preserve full experimental history and enable future analysis.

Next steps: Run archive_experiments.py to complete consolidation
EOF
echo "---"

echo ""
read -p "Commit these changes? (y/N): " commit_changes

if [[ $commit_changes =~ ^[Yy]$ ]]; then
    echo "💾 Committing consolidation documentation..."
    git commit -m "feat: Model consolidation documentation and strategy

Add comprehensive model consolidation framework:

- model_consolidation_report.md: Analysis of 23 experiments with performance metrics
- models_registry_consolidated.json: Production model registry  
- experiment_consolidation_log.json: Detailed move tracking and metadata
- git_consolidation_strategy.md: Git workflow for consolidation
- archive_experiments.py: Safe archiving script (no deletion)
- experiments/ARCHIVE_README.md: Archive directory documentation

Production models identified:
- baseline_v1: 58.2% accuracy, balanced F1 scores (RECOMMENDED)
- medium_v1: 56.3% accuracy, needs depression class tuning

Archive strategy: Move experiments to archive directories instead of deletion
to preserve full experimental history and enable future analysis.

Next steps: Run archive_experiments.py to complete consolidation"
    
    echo "✅ Committed successfully!"
    echo ""
    echo "📤 To push to remote:"
    echo "   git push -u origin model-consolidation"
else
    echo "⏸️  Changes staged but not committed"
fi

echo ""
echo "🎯 Next steps:"
echo "1. Review the committed consolidation documentation"
echo "2. Run 'python archive_experiments.py' to archive experiments"
echo "3. Push branch: 'git push -u origin model-consolidation'"
echo "4. Create PR to merge consolidation changes"
echo "5. Production models are in models/production/"

echo ""
echo "📊 Summary of consolidation:"
echo "  📁 Total experiments analyzed: 23"
echo "  🏆 Production models: 2 (moved to models/production/)"
echo "  📦 To be archived: ~19 (moved to experiments/archive_*/)"
echo "  🗑️  To be deleted: 0 (archive strategy preserves all experiments)"
echo "  📝 Full documentation: model_consolidation_report.md"
echo ""
echo "✅ Git consolidation workflow complete!"
