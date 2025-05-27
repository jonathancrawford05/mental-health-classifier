import subprocess
import os

# Change directory and run test
os.chdir("/Users/family_crawfords/projects/claude-mcp/mental-health-classifier")

try:
    result = subprocess.run(["python", "test_trainer_fix.py"], 
                          capture_output=True, text=True, check=True)
    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
except subprocess.CalledProcessError as e:
    print(f"Test failed with return code: {e.returncode}")
    print("STDOUT:")
    print(e.stdout)
    print("STDERR:")
    print(e.stderr)
