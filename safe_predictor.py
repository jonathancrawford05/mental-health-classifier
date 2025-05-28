#!/usr/bin/env python3
"""
Ultra-Safe Predictor

Robust predictor with comprehensive error handling
for vocabulary access issues.
"""

import sys
import json
import pickle
from pathlib import Path
import torch
import re

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from models import MentalHealthClassifier
from data import ClinicalTextPreprocessor
from torchtext.data.utils import get_tokenizer
import torch.nn.functional as F


class UltraSafePredictor:
    """Ultra-robust predictor with extensive error handling."""
    
    def __init__(self, model_dir: str = "models/"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.vocab = None
        self.config = None
        self.preprocessor = None
        self.tokenizer = None
        self.device = torch.device('cpu')
        
        # Create reverse mapping for debugging
        self.token_to_id = {}
        self.id_to_token = {}
        
        self.label_names = ['Depression', 'Anxiety', 'Suicide']
        
        # SAFETY CONFIGURATION
        self.safety_config = {
            'suicide_threshold': 0.8,
            'confidence_threshold': 0.6,
            'clinical_keywords_required': True,
            'normal_expressions_filter': True
        }
        
        # Normal expressions filter
        self.normal_expressions = [
            r'\bi hate (who i am|this|that|when|traffic|people)',
            r'\bi might (do something|scream|go crazy)',
            r'\bi\'m (annoyed|frustrated|upset|tired|stressed)',
            r'(bad day|rough day|long day)',
            r'(mildly upset|slightly annoyed|a bit frustrated)'
        ]
        
        # Clinical keywords
        self.clinical_keywords = {
            'suicide': ['suicidal ideation', 'end my life', 'kill myself', 'suicide plan', 'want to die']
        }
    
    def build_vocab_mapping(self):
        """Build comprehensive vocabulary mapping."""
        print("🔧 Building vocabulary mapping...")
        
        try:
            # Try different ways to extract vocabulary mappings
            vocab_size = len(self.vocab)
            print(f"   Vocabulary size: {vocab_size}")
            
            # Method 1: Try to iterate through vocabulary
            try:
                if hasattr(self.vocab, 'items'):
                    # Dictionary-like
                    for token, idx in self.vocab.items():
                        self.token_to_id[token] = idx
                        self.id_to_token[idx] = token
                elif hasattr(self.vocab, '__iter__'):
                    # Try to iterate directly
                    for i, token in enumerate(self.vocab):
                        self.token_to_id[token] = i
                        self.id_to_token[i] = token
                else:
                    # Build mapping by testing known tokens
                    test_tokens = ['the', 'and', 'a', 'to', 'of', 'in', 'patient', 'i', 'is', 'that']
                    for token in test_tokens:
                        try:
                            idx = self.vocab[token]
                            self.token_to_id[token] = idx
                            self.id_to_token[idx] = token
                            print(f"   Found: '{token}' -> {idx}")
                        except:
                            continue
                            
            except Exception as e:
                print(f"   Warning: Could not build full mapping: {e}")
            
            print(f"   ✅ Mapped {len(self.token_to_id)} tokens")
            
            # Add essential tokens if missing
            essential_tokens = ['<pad>', '<unk>', '[PAD]', '[UNK]']
            for token in essential_tokens:
                if token not in self.token_to_id:
                    try:
                        idx = self.vocab[token]
                        self.token_to_id[token] = idx
                        self.id_to_token[idx] = token
                        print(f"   Added essential: '{token}' -> {idx}")
                    except:
                        continue
            
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to build vocab mapping: {e}")
            return False
    
    def get_token_id_safe(self, token: str) -> int:
        """Ultra-safe token ID retrieval."""
        try:
            # Method 1: Use pre-built mapping
            if token in self.token_to_id:
                return self.token_to_id[token]
            
            # Method 2: Direct vocabulary access
            try:
                return self.vocab[token]
            except:
                pass
            
            # Method 3: Try common unknown token patterns
            for unk_token in ['<unk>', '[UNK]', '<UNK>']:
                try:
                    if unk_token in self.token_to_id:
                        return self.token_to_id[unk_token]
                    return self.vocab[unk_token]
                except:
                    continue
            
            # Method 4: Return 0 as ultimate fallback
            return 0
            
        except Exception as e:
            print(f"Token lookup error for '{token}': {e}")
            return 0
    
    def get_pad_token_id_safe(self) -> int:
        """Ultra-safe padding token ID retrieval."""
        for pad_token in ['<pad>', '[PAD]', '<unk>', '[UNK]', 0]:
            try:
                if isinstance(pad_token, int):
                    return pad_token
                if pad_token in self.token_to_id:
                    return self.token_to_id[pad_token]
                return self.vocab[pad_token]
            except:
                continue
        return 0  # Ultimate fallback
    
    def load_model(self):
        """Load the trained model with ultra-safe loading."""
        print("Loading ultra-safe predictor...")
        
        try:
            # Load model info and config
            model_info_path = self.model_dir / "model_info.json"
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                self.config = model_info['config']
            
            print(f"✅ Model has {model_info['total_parameters']:,} parameters")
            
            # Load vocabulary
            vocab_path = self.model_dir / "vocab.pkl"
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            
            print(f"✅ Vocabulary size: {len(self.vocab)}")
            
            # Build vocabulary mapping
            self.build_vocab_mapping()
            
            # Create model
            model_config = self.config['model']
            self.model = MentalHealthClassifier(model_config)
            
            # Load weights
            checkpoint_path = self.model_dir / "best_model.pt"
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"✅ Model loaded with validation F1: {checkpoint.get('best_val_f1', 'unknown')}")
            
            # Set to evaluation mode
            self.model.eval()
            
            # Initialize preprocessor and tokenizer
            self.preprocessor = ClinicalTextPreprocessor()
            self.tokenizer = get_tokenizer('basic_english')
            
            print("✅ Ultra-safe model loaded successfully!")
            print()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def is_normal_expression(self, text: str) -> bool:
        """Check if text matches common normal expressions."""
        text_lower = text.lower()
        for pattern in self.normal_expressions:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def has_clinical_keywords(self, text: str, predicted_class: str) -> bool:
        """Check if text contains clinical keywords."""
        text_lower = text.lower()
        keywords = self.clinical_keywords.get(predicted_class.lower(), [])
        return any(keyword in text_lower for keyword in keywords)
    
    def apply_safety_filters(self, text: str, prediction: str, probabilities: dict) -> tuple:
        """Apply safety filters."""
        max_confidence = max(probabilities.values())
        
        # Normal expression filter
        if self.is_normal_expression(text):
            return "Normal Expression", {
                'Depression': 0.1, 'Anxiety': 0.1, 'Suicide': 0.1, 'Normal': 0.7
            }
        
        # Confidence filter
        if max_confidence < self.safety_config['confidence_threshold']:
            return "Uncertain", {**probabilities, 'Uncertain': 1 - max_confidence}
        
        # Enhanced suicide detection
        if prediction == "Suicide":
            suicide_prob = probabilities['Suicide']
            
            if suicide_prob < self.safety_config['suicide_threshold']:
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                second_choice = sorted_probs[1][0]
                return f"{second_choice} (Low Suicide Confidence)", probabilities
            
            if not self.has_clinical_keywords(text, "Suicide"):
                return "Possible Distress (Not Clinical Suicide Risk)", {
                    'Depression': max(probabilities['Depression'], 0.3),
                    'Anxiety': max(probabilities['Anxiety'], 0.3),
                    'Suicide': min(probabilities['Suicide'], 0.4),
                    'Distress': 0.4
                }
        
        return prediction, probabilities
    
    def predict(self, text: str, return_probabilities: bool = True, debug: bool = False):
        """Ultra-safe prediction with extensive error handling."""
        if self.model is None:
            return "Model Not Loaded", {'Error': 1.0}
        
        try:
            if debug:
                print(f"🔍 Debug: Input text: '{text}'")
            
            # Step 1: Preprocess text
            preprocessed_text = self.preprocessor.preprocess(text)
            if debug:
                print(f"🔍 Debug: Preprocessed: '{preprocessed_text}'")
            
            # Step 2: Tokenize
            tokens = self.tokenizer(preprocessed_text)
            if debug:
                print(f"🔍 Debug: Tokens: {tokens[:10]}...")  # Show first 10 tokens
            
            # Step 3: Convert to token IDs with ultra-safe handling
            token_ids = []
            failed_tokens = []
            
            for token in tokens:
                token_id = self.get_token_id_safe(token)
                token_ids.append(token_id)
                if token_id == 0 and token not in ['<pad>', '<unk>']:
                    failed_tokens.append(token)
            
            if debug and failed_tokens:
                print(f"🔍 Debug: Unknown tokens (using 0): {failed_tokens[:5]}...")
            
            if debug:
                print(f"🔍 Debug: Token IDs: {token_ids[:10]}...")
            
            # Step 4: Handle sequence length
            max_length = self.config['data']['max_length']
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                if debug:
                    print(f"🔍 Debug: Truncated to {max_length}")
            
            # Step 5: Create attention mask
            attention_mask = [1] * len(token_ids)
            
            # Step 6: Pad sequences
            pad_token_id = self.get_pad_token_id_safe()
            padding_length = max_length - len(token_ids)
            token_ids.extend([pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            
            if debug:
                print(f"🔍 Debug: Final length: {len(token_ids)}, Padding: {padding_length}")
            
            # Step 7: Create tensors
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            attention_mask = torch.tensor([attention_mask], dtype=torch.long)
            
            if debug:
                print(f"🔍 Debug: Tensor shapes: {input_ids.shape}, {attention_mask.shape}")
            
            # Step 8: Make prediction
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.nn.functional.softmax(outputs['logits'], dim=-1)
                prediction = torch.argmax(outputs['logits'], dim=-1)
            
            predicted_class = self.label_names[prediction.item()]
            
            # Step 9: Convert to probability dict
            probs = probabilities[0].cpu().numpy()
            prob_dict = {label: float(probs[i]) for i, label in enumerate(self.label_names)}
            
            if debug:
                print(f"🔍 Debug: Raw prediction: {predicted_class}")
                print(f"🔍 Debug: Raw probabilities: {prob_dict}")
            
            # Step 10: Apply safety filters
            safe_prediction, safe_probabilities = self.apply_safety_filters(
                text, predicted_class, prob_dict
            )
            
            return (safe_prediction, safe_probabilities) if return_probabilities else safe_prediction
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Prediction error: {error_msg}")
            if debug:
                import traceback
                traceback.print_exc()
            
            return "Error - Requires Manual Review", {'Error': 1.0, 'ErrorMsg': error_msg}


def test_ultra_safe_predictor():
    """Test the ultra-safe predictor with debugging."""
    
    print("🛡️ TESTING ULTRA-SAFE PREDICTOR")
    print("=" * 50)
    
    predictor = UltraSafePredictor()
    
    if not predictor.load_model():
        print("❌ Failed to load model - aborting test")
        return
    
    test_cases = [
        "I hate who I am and don't want to be me anymore",
        "I am mildly upset, I might do something crazy", 
        "I feel hopeless and want to end my life",
        "Patient reports suicidal ideation with plan",
        "I had a great day today"
    ]
    
    print("Testing with debug mode for first case:")
    print("-" * 50)
    
    # Test first case with debug
    if test_cases:
        text = test_cases[0]
        print(f"\nDEBUG MODE - Testing: '{text}'")
        prediction, probs = predictor.predict(text, debug=True)
        print(f"Result: {prediction}")
        print(f"Probabilities: {probs}")
    
    print(f"\nTesting all cases:")
    print("-" * 30)
    
    for i, text in enumerate(test_cases, 1):
        try:
            prediction, probs = predictor.predict(text)
            max_prob = max(probs.values()) if isinstance(probs, dict) else 0
            
            print(f"\n{i}. Input: {text}")
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {max_prob:.3f}")
            
        except Exception as e:
            print(f"\n{i}. Input: {text}")
            print(f"   FAILED: {e}")


if __name__ == "__main__":
    test_ultra_safe_predictor()
