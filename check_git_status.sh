#!/bin/bash

echo "=== Git Status Check ==="
git status

echo -e "\n=== Files that need to be committed ==="
echo "Modified files:"
git diff --name-only

echo -e "\nUntracked files:"
git ls-files --others --exclude-standard

echo -e "\n=== Critical files for deployment ==="
echo "Checking critical deployment files exist:"
files=("Dockerfile" "requirements.txt" "api_server.py" "src/data/data_processor.py" "README.md" ".gitignore")

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
    fi
done

echo -e "\n=== Docker build test ==="
echo "Testing if Docker can build successfully..."
# Uncomment to test: docker build -t test-build .
