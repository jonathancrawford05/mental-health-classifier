#!/usr/bin/env python3
"""
Test runner script that properly sets up the Python path.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now run the tests
if __name__ == "__main__":
    print("Setting up Python path...")
    print(f"Project root: {project_root}")
    print(f"Source path: {src_path}")
    print(f"Python path: {sys.path[:3]}...")  # Show first few entries
    
    # Import and run tests
    try:
        from tests.test_basic_fixed import run_all_tests
        run_all_tests()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Let's run the tests directly...")
        
        # Run tests directly
        exec(open("tests/test_basic_fixed.py").read())
