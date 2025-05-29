#!/usr/bin/env python3
"""
Simple Ultra-Safe Predictor

Compatible with the new 20K model format - simplified to avoid syntax errors.
"""

import sys
import json
import pickle
from pathlib import Path
import torch

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from models import MentalHealthClassifier
from data import ClinicalTextPreprocessor
from torchtext.data.utils import get_tokenizer
import torch.nn.functional as F


class UltraSafePredictor:
    """Ultra-robust predictor compatible with 20K model format."""
    
    def __init__(self, model_dir: str = "models/"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.vocab = None
        self.config = None
        self.model_config = None
        self.preprocessor = None
        self.tokenizer = None
        self.device = torch.device('cpu')
        
        # Create reverse mapping for debugging
        self.token_to_id = {}
        self.id_to_token = {}
        
        self.label_names = ['Anxiety', 'Depression', 'Suicide']  # Updated order from 20K model
        
        # SAFETY CONFIGURATION
        self.safety_config = {
            'suicide_threshold': 0.7,
            'confidence_threshold': 0.6,
        }
        
        # Normal expressions (simplified - no regex)
        self.normal_phrases = [
            "i hate traffic",
            "bad day",
            "rough day", 
            "long day",
            "mildly upset",
            "slightly annoyed",
            "a bit frustrated",
            "this is killing me",
            "i could just die",
            "dead tired"
        ]
        
        # Clinical keywords
        self.clinical_keywords = {
            'suicide': ['suicidal ideation', 'end my life', 'kill myself', 'suicide plan', 'want to die', 'not want to live']
        }
    
    def load_model(self):
        """Load the 20K trained model with compatible format handling."""
        print("Loading ultra-safe predictor...")
        
        try:
            # Try to load from model checkpoint first (new format)
            checkpoint_path = self.model_dir / "best_model.pt"
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Check if it's the new format
            if 'config' in checkpoint:
                self.model_config = checkpoint['config']
                print("✅ Loaded config from model checkpoint")
            else:
                # Fall back to model_info.json
                model_info_path = self.model_dir / "model_info.json"
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                    self.model_config = model_info['model_config']  # Note: model_config not config
                print("✅ Loaded config from model info")
            
            print(f"✅ Model architecture: {self.model_config['n_layer']}L-{self.model_config['n_embd']}D-{self.model_config['num_heads']}H")
            
            # Load vocabulary
            vocab_path = self.model_dir / "vocab.pkl"
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            
            print(f"✅ Vocabulary size: {len(self.vocab)}")
            
            # Build vocabulary mapping
            self.build_vocab_mapping()
            
            # Create model with the loaded config
            self.model = MentalHealthClassifier(self.model_config)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Get performance info
            val_acc = checkpoint.get('best_val_accuracy', checkpoint.get('best_val_f1', 'unknown'))
            test_acc = checkpoint.get('test_accuracy', 'unknown')
            print(f"✅ Model performance: Val={val_acc}, Test={test_acc}")
            
            # Set to evaluation mode
            self.model.eval()
            
            # Initialize preprocessor and tokenizer
            self.preprocessor = ClinicalTextPreprocessor()
            self.tokenizer = get_tokenizer('basic_english')
            
            print("✅ Ultra-safe predictor loaded successfully!")
            print()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def build_vocab_mapping(self):
        """Build comprehensive vocabulary mapping."""
        print("🔧 Building vocabulary mapping...")
        
        try:
            vocab_size = len(self.vocab)
            print(f"   Vocabulary size: {vocab_size}")
            
            # Handle vocabulary mapping
            if hasattr(self.vocab, 'items'):
                # Dictionary-like vocabulary
                for token, idx in self.vocab.items():
                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token
            elif hasattr(self.vocab, '__getitem__') and hasattr(self.vocab, '__iter__'):
                # Try to iterate and access
                try:
                    for token in list(self.vocab)[:100]:  # Test first 100
                        try:
                            idx = self.vocab[token]
                            self.token_to_id[token] = idx
                            self.id_to_token[idx] = token
                        except:
                            continue
                except:
                    pass
            
            print(f"   ✅ Mapped {len(self.token_to_id)} token-to-id pairs")
            
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
            
            # Method 4: Return 1 (common UNK ID)
            return 1
            
        except Exception as e:
            return 1  # Safe fallback
    
    def get_pad_token_id_safe(self) -> int:
        """Ultra-safe padding token ID retrieval."""
        for pad_token in ['<pad>', '[PAD]', '<unk>', '[UNK]']:
            try:
                if pad_token in self.token_to_id:
                    return self.token_to_id[pad_token]
                return self.vocab[pad_token]
            except:
                continue
        return 0  # Ultimate fallback
    
    def is_normal_expression(self, text: str) -> bool:
        """Check if text matches common normal expressions (simplified)."""
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in self.normal_phrases)
    
    def has_clinical_keywords(self, text: str, predicted_class: str) -> bool:
        """Check if text contains clinical keywords."""
        text_lower = text.lower()
        keywords = self.clinical_keywords.get(predicted_class.lower(), [])
        return any(keyword in text_lower for keyword in keywords)
    
    def apply_safety_filters(self, text: str, prediction: str, probabilities: dict) -> tuple:
        """Apply safety filters."""
        max_confidence = max(probabilities.values()) if probabilities else 0
        
        # Normal expression filter
        if self.is_normal_expression(text):
            return "Normal Expression", {
                'Anxiety': 0.1, 'Depression': 0.1, 'Suicide': 0.1, 'Normal': 0.7
            }
        
        # Confidence filter
        if max_confidence < self.safety_config['confidence_threshold']:
            return "Uncertain", {**probabilities, 'Uncertain': 1 - max_confidence}
        
        # Enhanced suicide detection
        if prediction == "Suicide":
            suicide_prob = probabilities['Suicide']
            
            if suicide_prob < self.safety_config['suicide_threshold']:
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                second_choice = sorted_probs[1][0] if len(sorted_probs) > 1 else "Depression"
                return f"{second_choice} (Low Suicide Confidence)", probabilities
            
            if not self.has_clinical_keywords(text, "Suicide"):
                return "Possible Distress (Not Clinical Suicide Risk)", {
                    'Anxiety': max(probabilities.get('Anxiety', 0), 0.3),
                    'Depression': max(probabilities.get('Depression', 0), 0.3),
                    'Suicide': min(probabilities.get('Suicide', 0), 0.4),
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
                print(f"🔍 Debug: Tokens: {tokens[:10]}...")
            
            # Step 3: Convert to token IDs
            token_ids = []
            failed_tokens = []
            
            for token in tokens:
                token_id = self.get_token_id_safe(token)
                token_ids.append(token_id)
                if token_id in [0, 1] and token not in ['<pad>', '<unk>', 'the', 'a']:
                    failed_tokens.append(token)
            
            if debug and failed_tokens:
                print(f"🔍 Debug: Unknown tokens: {failed_tokens[:5]}...")
            
            # Step 4: Handle sequence length
            max_length = self.model_config['max_seq_length']
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
            attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long)
            
            # Step 8: Make prediction
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask_tensor)
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


def test_fixed_predictor():
    """Test the fixed predictor."""
    
    print("🛡️ TESTING SIMPLE ULTRA-SAFE PREDICTOR")
    print("=" * 50)
    
    predictor = UltraSafePredictor()
    
    if not predictor.load_model():
        print("❌ Failed to load model - aborting test")
        return
    
    test_cases = [
        "I feel hopeless and want to end my life",
        "I hate traffic and this long day",
        "Patient reports suicidal ideation with plan", 
        "I'm feeling really anxious about work",
        "Everything feels meaningless and dark"
    ]
    
    print(f"Testing {len(test_cases)} cases:")
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
    
    print("\n✅ Simple predictor test complete!")


if __name__ == "__main__":
    test_fixed_predictor()
