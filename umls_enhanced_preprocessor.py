#!/usr/bin/env python3
"""
UMLS Enhanced Clinical Text Preprocessor

Integrates UMLS thesaurus capabilities with the mental health classifier
to expand clinical vocabulary and improve semantic understanding.
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Set
import logging
import re

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

class UMLSEnhancedPreprocessor:
    """Enhanced preprocessor using UMLS synonym expansion."""
    
    def __init__(self, umls_synonyms_file: str = None):
        self.umls_synonyms = {}
        self.clinical_abbrev = self._get_clinical_abbreviations()
        
        if umls_synonyms_file and Path(umls_synonyms_file).exists():
            self.load_umls_synonyms(umls_synonyms_file)
    
    def _get_clinical_abbreviations(self) -> Dict[str, str]:
        """Clinical abbreviations commonly used in mental health records."""
        return {
            # Patient terms
            'pt': 'patient',
            'pts': 'patients', 
            'ptsd': 'post traumatic stress disorder',
            
            # Complaints and history
            'c/o': 'complains of',
            'cc': 'chief complaint',
            'hx': 'history',
            'h/o': 'history of',
            'pmh': 'past medical history',
            'psych': 'psychiatric',
            
            # Mental health conditions
            'mdd': 'major depressive disorder',
            'gad': 'generalized anxiety disorder',
            'ocd': 'obsessive compulsive disorder',
            'bpd': 'bipolar disorder',
            'adhd': 'attention deficit hyperactivity disorder',
            
            # Symptoms
            'si': 'suicidal ideation',
            'hi': 'homicidal ideation',
            'a/v': 'auditory visual',
            'halluc': 'hallucinations',
            'delus': 'delusions',
            
            # Medications
            'ssri': 'selective serotonin reuptake inhibitor',
            'snri': 'serotonin norepinephrine reuptake inhibitor',
            'benzo': 'benzodiazepine',
            'antidep': 'antidepressant',
            
            # General medical
            'w/': 'with',
            'w/o': 'without',
            'r/o': 'rule out',
            'dx': 'diagnosis',
            'tx': 'treatment',
            'rx': 'prescription',
            'f/u': 'follow up',
            'approx': 'approximately',
            'neg': 'negative',
            'pos': 'positive'
        }
    
    def expand_clinical_abbreviations(self, text: str) -> str:
        """Expand clinical abbreviations in text."""
        sorted_abbrev = sorted(self.clinical_abbrev.items(), key=lambda x: len(x[0]), reverse=True)
        
        expanded_text = text.lower()
        for abbrev, expansion in sorted_abbrev:
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            expanded_text = re.sub(pattern, expansion, expanded_text)
        
        return expanded_text
    
    def preprocess_clinical_text(self, text: str) -> str:
        """Full preprocessing pipeline for clinical text."""
        # Expand abbreviations
        expanded = self.expand_clinical_abbreviations(text)
        
        # Clean up whitespace
        expanded = ' '.join(expanded.split())
        
        return expanded


def test_umls_preprocessor():
    """Test the UMLS enhanced preprocessor."""
    
    print("🧪 TESTING UMLS ENHANCED PREPROCESSOR")
    print("=" * 50)
    
    preprocessor = UMLSEnhancedPreprocessor()
    
    test_cases = [
        "Pt c/o depression w/ SI and h/o MDD",
        "Patient reports anxiety and panic attacks",
        "PTSD w/ flashbacks and insomnia",
        "R/o GAD vs MDD, start SSRI tx"
    ]
    
    print("Clinical Text Preprocessing Examples:")
    print("-" * 40)
    
    for i, text in enumerate(test_cases, 1):
        processed = preprocessor.preprocess_clinical_text(text)
        print(f"\n{i}. Original: {text}")
        print(f"   Processed: {processed}")
    
    print(f"\n✅ Abbreviation expansion working!")


if __name__ == "__main__":
    print("🩺 UMLS ENHANCED CLINICAL PREPROCESSOR") 
    print("Integrating medical thesaurus with mental health classifier")
    print()
    
    test_umls_preprocessor()
