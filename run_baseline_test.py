import subprocess
import os

# Change to the mental-health-classifier directory
os.chdir("/Users/family_crawfords/projects/claude-mcp/mental-health-classifier")

# Run the baseline experiment
cmd = [
    "python", "train_with_tracking.py",
    "--experiment-config", "baseline_small", 
    "--create-sample-data",
    "--sample-size", "1000",
    "--seed", "42"
]

print("🧪 Running baseline experiment...")
print(f"Command: {' '.join(cmd)}")
print("=" * 60)

try:
    # Run the command and capture output in real time
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              universal_newlines=True, bufsize=1)
    
    # Print output in real time
    for line in process.stdout:
        print(line.strip())
    
    # Wait for completion
    process.wait()
    
    if process.returncode == 0:
        print("\n✅ Baseline experiment completed successfully!")
    else:
        print(f"\n❌ Experiment failed with return code: {process.returncode}")
        
except Exception as e:
    print(f"Error running experiment: {e}")
