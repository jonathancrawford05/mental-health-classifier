#!/usr/bin/env python3
"""
Model Inference Script

This script shows how to load and use the trained mental health classifier
for making predictions on new text inputs.
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


class MentalHealthPredictor:
    """Class for loading and using trained mental health classifier."""
    
    def __init__(self, model_dir: str = "models/"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.vocab = None
        self.config = None
        self.preprocessor = None
        self.tokenizer = None
        self.device = torch.device('cpu')  # Use CPU for inference
        
        self.label_names = ['Depression', 'Anxiety', 'Suicide']
        
    def load_model(self):
        """Load the trained model and all required components."""
        print("Loading trained mental health classifier...")
        
        # Load model info and config
        model_info_path = self.model_dir / "model_info.json"
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
            self.config = model_info['config']
        
        print(f"Model has {model_info['total_parameters']:,} parameters")
        
        # Load vocabulary
        vocab_path = self.model_dir / "vocab.pkl"
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        print(f"Vocabulary size: {len(self.vocab)}")
        
        # Create model with same config as training
        model_config = self.config['model']
        self.model = MentalHealthClassifier(model_config)
        
        # Load trained weights
        checkpoint_path = self.model_dir / "best_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded model with validation F1: {checkpoint.get('best_val_f1', 'unknown')}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize preprocessor and tokenizer
        self.preprocessor = ClinicalTextPreprocessor()
        self.tokenizer = get_tokenizer('basic_english')
        
        print("✅ Model loaded successfully!\n")
    
    def predict(self, text: str, return_probabilities: bool = True):
        """Make prediction on input text."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess text
        preprocessed_text = self.preprocessor.preprocess(text)
        tokens = self.tokenizer(preprocessed_text)
        token_ids = [self.vocab[token] for token in tokens]
        
        # Use max_length from training config
        max_length = self.config['data']['max_length']
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        # Pad to max_length
        pad_token_id = self.vocab['<pad>']
        padding_length = max_length - len(token_ids)
        token_ids.extend([pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        
        # Convert to tensors
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = F.softmax(outputs['logits'], dim=-1)
            prediction = torch.argmax(outputs['logits'], dim=-1)
        
        predicted_class = self.label_names[prediction.item()]
        
        if return_probabilities:
            probs = probabilities[0].cpu().numpy()
            prob_dict = {}
            for i, label in enumerate(self.label_names):
                prob_dict[label] = float(probs[i])
            return predicted_class, prob_dict
        
        return predicted_class
    
    def predict_batch(self, texts: list):
        """Make predictions on multiple texts."""
        results = []
        for text in texts:
            prediction, probabilities = self.predict(text, return_probabilities=True) 
            results.append({
                'text': text,
                'prediction': prediction,
                'probabilities': probabilities
            })
        return results


def demo_predictions():
    """Demonstrate how to use the trained model."""
    print("=" * 80)
    print("MENTAL HEALTH CLASSIFIER - INFERENCE DEMO")
    print("=" * 80)
    
    # Initialize predictor
    predictor = MentalHealthPredictor()
    
    # Load the trained model
    predictor.load_model()
    
    # Test examples
    test_examples = [
        "I feel hopeless and can't see any way out of this situation",
        "I'm constantly worried about everything that could go wrong", 
        "I've been having thoughts about ending my life",
        "Patient reports feeling great and enjoying daily activities",
        "Pt c/o severe depression w/ SI and h/o anxiety",  # Clinical text
        "Unable to sleep, no appetite, feeling worthless every day"
    ]
    
    print("Making predictions on sample texts:\n")
    
    for i, text in enumerate(test_examples, 1):
        print(f"Example {i}:")
        print(f"Text: {text}")
        
        # Get prediction
        prediction, probabilities = predictor.predict(text)
        
        print(f"Predicted: {prediction}")
        print("Probabilities:")
        for label, prob in probabilities.items():
            print(f"  {label}: {prob:.3f}")
        print("-" * 60)
    
    print("\n🔬 Clinical Text Processing Demo:")
    clinical_text = "Pt c/o depression w/ SI and h/o MDD"
    preprocessed = predictor.preprocessor.preprocess(clinical_text)
    print(f"Original: {clinical_text}")
    print(f"Processed: {preprocessed}")
    
    prediction, probs = predictor.predict(clinical_text)
    print(f"Prediction: {prediction}")
    print(f"Confidence: {max(probs.values()):.3f}")


def interactive_mode():
    """Interactive mode for testing custom text."""
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("Enter text to classify (or 'quit' to exit):\n")
    
    predictor = MentalHealthPredictor()
    predictor.load_model()
    
    while True:
        try:
            user_input = input("Enter text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
                
            prediction, probabilities = predictor.predict(user_input)
            
            print(f"\nPrediction: {prediction}")
            print("Confidence scores:")
            for label, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                print(f"  {label}: {prob:.3f}")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mental Health Classifier Inference")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--text", "-t", type=str,
                       help="Classify a single text")
    
    args = parser.parse_args()
    
    if args.text:
        # Single prediction mode
        predictor = MentalHealthPredictor()
        predictor.load_model()
        prediction, probs = predictor.predict(args.text)
        print(f"Text: {args.text}")
        print(f"Prediction: {prediction}")
        for label, prob in probs.items():
            print(f"  {label}: {prob:.3f}")
    elif args.interactive:
        # Interactive mode
        interactive_mode()
    else:
        # Demo mode
        demo_predictions()
