#!/usr/bin/env python3
"""
Test 4-Class Model

Test the new 4-class model with the same edge cases to compare performance.
"""

import sys
from pathlib import Path
import json
import torch
import pickle
from edge_case_testing_complete import create_challenging_edge_cases

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from models import MentalHealthClassifier
from data import ClinicalTextPreprocessor
from torchtext.data.utils import get_tokenizer

class FourClassPredictor:
    """Predictor for 4-class model."""
    
    def __init__(self, model_dir: str = "models/"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.vocab = None
        self.config = None
        self.preprocessor = None
        self.tokenizer = None
        self.device = torch.device('cpu')
        self.label_names = ['Anxiety', 'Depression', 'Suicide', 'Normal']
    
    def load_model(self):
        """Load the 4-class model."""
        print("Loading 4-class predictor...")
        
        try:
            # Load model checkpoint
            checkpoint_path = self.model_dir / "best_model.pt"
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Get config
            if 'config' in checkpoint:
                self.config = checkpoint['config']
            else:
                model_info_path = self.model_dir / "model_info.json"
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                    self.config = model_info['model_config']
            
            print(f"✅ Model: {self.config['num_classes']} classes, {self.config['n_layer']} layers")
            
            # Load vocabulary
            vocab_path = self.model_dir / "vocab.pkl"
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            
            print(f"✅ Vocabulary: {len(self.vocab):,} tokens")
            
            # Create and load model
            self.model = MentalHealthClassifier(self.config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Initialize preprocessor
            self.preprocessor = ClinicalTextPreprocessor()
            self.tokenizer = get_tokenizer('basic_english')
            
            # Update label names if available
            if 'class_names' in checkpoint:
                self.label_names = checkpoint['class_names']
            
            print(f"✅ Classes: {self.label_names}")
            print("✅ 4-class predictor loaded successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def predict(self, text: str):
        """Make prediction on text."""
        if self.model is None:
            return "Error", {}
        
        try:
            # Preprocess
            preprocessed_text = self.preprocessor.preprocess(text)
            tokens = self.tokenizer(preprocessed_text)
            
            # Convert to IDs
            token_ids = []
            for token in tokens:
                if hasattr(self.vocab, 'get'):
                    token_id = self.vocab.get(token, self.vocab.get('<unk>', 1))
                else:
                    try:
                        token_id = self.vocab[token]
                    except:
                        token_id = 1
                token_ids.append(token_id)
            
            # Handle sequence length
            max_length = self.config['max_seq_length']
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            # Pad
            padding = [0] * (max_length - len(token_ids))
            token_ids.extend(padding)
            attention_mask = [1] * len(tokens) + [0] * len(padding)
            attention_mask = attention_mask[:max_length]
            
            # Create tensors
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask_tensor)
                probabilities = torch.nn.functional.softmax(outputs['logits'], dim=-1)
                prediction = torch.argmax(outputs['logits'], dim=-1)
            
            predicted_class = self.label_names[prediction.item()]
            probs = probabilities[0].cpu().numpy()
            prob_dict = {label: float(probs[i]) for i, label in enumerate(self.label_names)}
            
            return predicted_class, prob_dict
            
        except Exception as e:
            return f"Error: {str(e)}", {}

def test_4class_model():
    """Test the 4-class model on edge cases."""
    
    print("🧪 TESTING 4-CLASS MODEL")
    print("=" * 40)
    
    # Initialize predictor
    predictor = FourClassPredictor()
    
    if not predictor.load_model():
        print("❌ Failed to load 4-class model")
        return []
    
    # Get edge cases
    edge_cases = create_challenging_edge_cases()
    print(f"\\n📋 Testing {len(edge_cases)} challenging edge cases...")
    
    # Test each case
    results = []
    class_names = ['Anxiety', 'Depression', 'Suicide', 'Normal']
    
    print("\\n🔍 Running tests:")
    print("-" * 40)
    
    for i, case in enumerate(edge_cases, 1):
        text = case['text']
        true_label = case['label']
        true_class = class_names[true_label]
        category = case['category']
        
        try:
            # Get prediction
            predicted_class, probabilities = predictor.predict(text)
            
            # Map prediction to label
            if predicted_class in class_names:
                predicted_label = class_names.index(predicted_class)
            else:
                predicted_label = 3  # Default to Normal
            
            correct = predicted_label == true_label
            confidence = max(probabilities.values()) if probabilities else 0
            
            # Store result
            result = {
                'text': text,
                'true_class': true_class,
                'predicted_class': predicted_class,
                'correct': correct,
                'confidence': confidence,
                'category': category,
                'probabilities': probabilities
            }
            results.append(result)
            
            # Print result
            status = "✅" if correct else "❌"
            print(f"{i:2d}. {status} {true_class} -> {predicted_class} ({confidence:.3f})")
            if not correct:
                print(f"    Text: {text[:60]}...")
            
        except Exception as e:
            print(f"{i:2d}. ❌ ERROR: {str(e)}")
            results.append({
                'text': text,
                'true_class': true_class,
                'predicted_class': 'ERROR',
                'correct': False,
                'confidence': 0.0,
                'category': category,
                'probabilities': {}
            })
    
    # Analyze results
    print("\\n📊 4-CLASS MODEL ANALYSIS")
    print("=" * 30)
    
    total_cases = len(results)
    correct_cases = sum(1 for r in results if r['correct'])
    accuracy = correct_cases / total_cases if total_cases > 0 else 0
    
    print(f"Overall Accuracy: {accuracy:.3f} ({correct_cases}/{total_cases})")
    
    # Critical metrics
    suicide_cases = [r for r in results if r['true_class'] == 'Suicide']
    suicide_correct = sum(1 for r in suicide_cases if r['correct'])
    suicide_recall = suicide_correct / len(suicide_cases) if suicide_cases else 0
    
    false_suicide = [r for r in results if r['predicted_class'] == 'Suicide' and r['true_class'] != 'Suicide']
    false_alarm_rate = len(false_suicide) / total_cases if total_cases > 0 else 0
    
    normal_cases = [r for r in results if r['true_class'] == 'Normal']
    normal_correct = sum(1 for r in normal_cases if r['correct'])
    normal_precision = normal_correct / len(normal_cases) if normal_cases else 0
    
    print(f"\\n🎯 KEY METRICS:")
    print(f"   Suicide Recall: {suicide_recall:.3f} (was 0.75, target ≥0.85)")
    print(f"   False Alarm Rate: {false_alarm_rate:.3f} (was 0.357, target ≤0.15)")
    print(f"   Normal Precision: {normal_precision:.3f} (new metric)")
    
    # Deployment assessment
    suicide_safe = suicide_recall >= 0.85
    false_alarm_ok = false_alarm_rate <= 0.15
    overall_ok = accuracy >= 0.75
    
    print(f"\\n🚀 DEPLOYMENT READINESS:")
    print(f"   Suicide Detection: {'✅ PASS' if suicide_safe else '❌ FAIL'} ({suicide_recall:.1%})")
    print(f"   False Alarm Rate: {'✅ PASS' if false_alarm_ok else '❌ FAIL'} ({false_alarm_rate:.1%})")
    print(f"   Overall Accuracy: {'✅ PASS' if overall_ok else '❌ FAIL'} ({accuracy:.1%})")
    
    deployment_ready = suicide_safe and false_alarm_ok and overall_ok
    print(f"\\n🎯 STATUS: {'🟢 READY' if deployment_ready else '🔴 NEEDS WORK'}")
    
    # Show improvements
    print(f"\\n📈 IMPROVEMENTS vs 3-CLASS:")
    print(f"   Suicide Recall: 75.0% → {suicide_recall:.1%}")
    print(f"   False Alarms: 35.7% → {false_alarm_rate:.1%}")
    print(f"   Overall Accuracy: 23.8% → {accuracy:.1%}")
    
    # Save results
    with open('4class_test_results.json', 'w') as f:
        json.dump({
            'summary': {
                'accuracy': accuracy,
                'suicide_recall': suicide_recall,
                'false_alarm_rate': false_alarm_rate,
                'deployment_ready': deployment_ready
            },
            'results': results
        }, f, indent=2)
    
    print(f"\\n💾 Results saved to: 4class_test_results.json")
    
    return results

if __name__ == "__main__":
    results = test_4class_model()
    print(f"\\n🎯 4-class testing complete!")
