#!/usr/bin/env python3
"""
Quick debug script to test the clinical text preprocessing.
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from data import ClinicalTextPreprocessor
import re

def debug_clinical_preprocessing():
    """Debug the clinical text preprocessing."""
    print("=== DEBUGGING CLINICAL TEXT PREPROCESSING ===")
    
    preprocessor = ClinicalTextPreprocessor()
    
    # Test the w/ expansion specifically
    test_cases = [
        "w/ depression",
        "Patient w/ anxiety", 
        "Pt c/o depression w/ SI",
        "w/o symptoms",
        "patient w/ h/o depression"
    ]
    
    print("Clinical contractions dictionary:")
    for abbrev, expansion in preprocessor.clinical_contractions.items():
        print(f"  '{abbrev}' -> '{expansion}'")
    
    print("\nTesting regex patterns:")
    for abbrev, expansion in preprocessor.clinical_contractions.items():
        if abbrev in ["w/", "w/o", "c/o"]:
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            print(f"  Pattern for '{abbrev}': {pattern}")
    
    print("\nTesting expansions:")
    for test_text in test_cases:
        processed = preprocessor.preprocess(test_text)
        print(f"  '{test_text}' -> '{processed}'")
    
    print("\nTesting expand_clinical_contractions directly:")
    for test_text in test_cases:
        expanded = preprocessor.expand_clinical_contractions(test_text)
        print(f"  '{test_text}' -> '{expanded}'")
    
    # Test the specific regex pattern for w/
    print("\nTesting regex directly:")
    test_text = "patient w/ depression"
    pattern = r'\b' + re.escape("w/") + r'\b'
    result = re.sub(pattern, "with", test_text)
    print(f"  Pattern: {pattern}")
    print(f"  Text: '{test_text}' -> '{result}'")
    
    # Test if the issue is case sensitivity
    print("\nTesting case sensitivity:")
    test_lower = test_text.lower()
    result_lower = re.sub(pattern, "with", test_lower)
    print(f"  Lowercase: '{test_lower}' -> '{result_lower}'")

if __name__ == "__main__":
    debug_clinical_preprocessing()
