"""
Data processing utilities for mental health text classification.

Handles tokenization, vocabulary building, and dataset preparation
with clinical text preprocessing capabilities.
"""

import re
import string
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import List, Dict, Tuple, Optional, Iterator
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class ClinicalTextPreprocessor:
    """Preprocessor for clinical text with medical terminology handling."""
    
    def __init__(self, 
                 remove_stopwords: bool = False,
                 use_stemming: bool = False,
                 expand_contractions: bool = True,
                 normalize_clinical_terms: bool = True):
        
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.expand_contractions = expand_contractions
        self.normalize_clinical_terms = normalize_clinical_terms
        
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        if use_stemming:
            self.stemmer = PorterStemmer()
            
        # Clinical abbreviations and expansions
        self.clinical_contractions = {
            "pt": "patient",
            "hx": "history",
            "dx": "diagnosis", 
            "tx": "treatment",
            "sx": "symptoms",
            "rx": "prescription",
            "w/": "with",
            "w/o": "without",
            "c/o": "complains of",
            "s/p": "status post",
            "r/o": "rule out",
            "h/o": "history of",
            "f/u": "follow up",
            "sob": "shortness of breath",
            "cp": "chest pain",
            "n/v": "nausea and vomiting",
            "etoh": "alcohol",
        }
        
        # Mental health specific terms
        self.mental_health_terms = {
            "mdd": "major depressive disorder",
            "gad": "generalized anxiety disorder", 
            "ptsd": "post traumatic stress disorder",
            "ocd": "obsessive compulsive disorder",
            "adhd": "attention deficit hyperactivity disorder",
            "si": "suicidal ideation",
            "sa": "suicide attempt",
            "shi": "self harm ideation",
        }
    
    def expand_clinical_contractions(self, text: str) -> str:
        """Expand clinical abbreviations and contractions."""
        text = text.lower()
        
        # Handle special cases with slashes first (w/, w/o, c/o)
        # These need special handling because / is not a word boundary character
        slash_contractions = {
            "w/": "with",
            "w/o": "without", 
            "c/o": "complains of",
            "s/p": "status post",
            "h/o": "history of",
            "f/u": "follow up",
            "r/o": "rule out"
        }
        
        for abbrev, expansion in slash_contractions.items():
            # Use word boundaries but handle the slash specially
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text)
            # Also try without word boundary at the end for slash terms
            pattern_alt = r'\b' + re.escape(abbrev) + r'(?=\s|$)'
            text = re.sub(pattern_alt, expansion, text)
            
        # Handle regular abbreviations (no slashes)
        regular_contractions = {
            "pt": "patient",
            "hx": "history",
            "dx": "diagnosis", 
            "tx": "treatment",
            "sx": "symptoms",
            "rx": "prescription",
            "sob": "shortness of breath",
            "cp": "chest pain",
            "n/v": "nausea and vomiting",
            "etoh": "alcohol",
        }
        
        for abbrev, expansion in regular_contractions.items():
            text = re.sub(r'\b' + re.escape(abbrev) + r'\b', expansion, text)
            
        # Expand mental health terms
        for abbrev, expansion in self.mental_health_terms.items():
            text = re.sub(r'\b' + re.escape(abbrev) + r'\b', expansion, text)
            
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize clinical text."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Expand contractions if enabled
        if self.expand_contractions:
            text = self.expand_clinical_contractions(text)
        
        # Remove special characters but keep medical notation
        text = re.sub(r'[^\w\s\-\./]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra spaces
        text = text.strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text with clinical preprocessing."""
        # Clean text first
        text = self.clean_text(text)
        
        # Basic tokenization
        tokens = text.split()
        
        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming if enabled
        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def preprocess(self, text: str) -> str:
        """Main preprocessing pipeline."""
        return self.clean_text(text)


class MentalHealthDataset(Dataset):
    """PyTorch Dataset for mental health text classification."""
    
    def __init__(self, 
                 texts: List[str], 
                 labels: List[int],
                 vocab: object,
                 tokenizer: callable,
                 max_length: int = 512,
                 pad_token_id: int = 0):
        
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize and convert to indices
        tokens = self.tokenizer(text)
        token_ids = [self.vocab[token] for token in tokens]
        
        # Truncate or pad to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(token_ids)
        
        # Pad to max_length
        padding_length = self.max_length - len(token_ids)
        token_ids.extend([self.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DataProcessor:
    """Main data processing class for mental health classification."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.preprocessor = ClinicalTextPreprocessor()
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = None
        
        # Label mapping
        self.label_map = {
            'depression': 0,
            'anxiety': 1, 
            'suicide': 2
        }
        
        self.label_names = ['Depression', 'Anxiety', 'Suicide']
    
    def yield_tokens(self, texts: Iterator[str]) -> Iterator[List[str]]:
        """Yield tokenized texts for vocabulary building."""
        for text in texts:
            preprocessed_text = self.preprocessor.preprocess(text)
            yield self.tokenizer(preprocessed_text)
    
    def build_vocabulary(self, texts: List[str], min_freq: int = 2) -> None:
        """Build vocabulary from training texts."""
        logging.info("Building vocabulary...")
        
        vocab = build_vocab_from_iterator(
            self.yield_tokens(texts),
            min_freq=min_freq,
            specials=['<unk>', '<pad>', '<sos>', '<eos>']
        )
        vocab.set_default_index(vocab['<unk>'])
        
        self.vocab = vocab
        logging.info(f"Vocabulary built with {len(vocab)} tokens")
    
    def encode_labels(self, labels: List[str]) -> List[int]:
        """Convert string labels to integer indices."""
        return [self.label_map.get(label.lower(), 0) for label in labels]
    
    def decode_labels(self, label_ids: List[int]) -> List[str]:
        """Convert integer label indices to string labels."""
        return [self.label_names[label_id] for label_id in label_ids]
    
    def load_data(self, file_path: str) -> Tuple[List[str], List[int]]:
        """Load data from CSV file."""
        df = pd.read_csv(file_path)
        
        text_column = self.config.get('text_column', 'text')
        label_column = self.config.get('label_column', 'label')
        
        texts = df[text_column].astype(str).tolist()
        labels = self.encode_labels(df[label_column].tolist())
        
        return texts, labels
    
    def create_dataset(self, texts: List[str], labels: List[int]) -> MentalHealthDataset:
        """Create PyTorch dataset from texts and labels."""
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        max_length = self.config.get('max_length', 512)
        pad_token_id = self.vocab['<pad>']
        
        return MentalHealthDataset(
            texts=texts,
            labels=labels,
            vocab=self.vocab,
            tokenizer=lambda x: self.tokenizer(self.preprocessor.preprocess(x)),
            max_length=max_length,
            pad_token_id=pad_token_id
        )
    
    def create_dataloaders(self, 
                          train_texts: List[str], train_labels: List[int],
                          val_texts: List[str], val_labels: List[int],
                          test_texts: Optional[List[str]] = None,
                          test_labels: Optional[List[int]] = None) -> Dict[str, DataLoader]:
        """Create data loaders for training, validation, and testing."""
        
        batch_size = self.config.get('batch_size', 32)
        
        # Create datasets
        train_dataset = self.create_dataset(train_texts, train_labels)
        val_dataset = self.create_dataset(val_texts, val_labels)
        
        # Set multiprocessing start method to avoid resource leaks
        import multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        # Use num_workers=0 to avoid multiprocessing issues with resource tracking
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=0, pin_memory=False, persistent_workers=False),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=False, persistent_workers=False)
        }
        
        if test_texts is not None and test_labels is not None:
            test_dataset = self.create_dataset(test_texts, test_labels)
            dataloaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                           num_workers=0, pin_memory=False, persistent_workers=False)
        
        return dataloaders
    
    def get_class_weights(self, labels: List[int]) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets with enhanced weighting for suicide class."""
        from collections import Counter
        import numpy as np
        
        label_counts = Counter(labels)
        total_samples = len(labels)
        num_classes = len(self.label_names)
        
        weights = []
        for i in range(num_classes):
            if i in label_counts:
                weight = total_samples / (num_classes * label_counts[i])
                # Give extra weight to suicide class (class 2) due to its clinical importance
                if i == 2:  # Suicide class
                    weight *= 2.0  # Double the weight for suicide class
            else:
                weight = 1.0
            weights.append(weight)
        
        print(f"Class weights calculated: Depression={weights[0]:.2f}, Anxiety={weights[1]:.2f}, Suicide={weights[2]:.2f}")
        return torch.FloatTensor(weights)
    
    def save_vocabulary(self, path: str) -> None:
        """Save vocabulary to file."""
        if self.vocab is None:
            raise ValueError("No vocabulary to save")
        
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.vocab, f)
        
        logging.info(f"Vocabulary saved to {path}")
    
    def load_vocabulary(self, path: str) -> None:
        """Load vocabulary from file."""
        import pickle
        with open(path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        logging.info(f"Vocabulary loaded from {path}")


def create_sample_data(output_path: str, num_samples: int = 1000) -> None:
    """Create sample dataset for testing purposes."""
    import random
    
    # Sample texts for each category
    depression_samples = [
        "I feel hopeless and empty inside, nothing brings me joy anymore",
        "Patient reports persistent sadness and loss of interest in activities",
        "Experiencing severe depression with suicidal thoughts daily",
        "Unable to sleep, no appetite, feeling worthless and guilty",
        "Major depressive episode with psychomotor retardation",
        "Chronic depression affecting work and relationships significantly",
        "Feeling isolated and disconnected from everyone around me",
        "Patient exhibits classic symptoms of major depressive disorder"
    ]
    
    anxiety_samples = [
        "Constant worry and fear about everything, can't relax",
        "Patient reports severe anxiety attacks with heart palpitations", 
        "Overwhelming panic and dread, feeling like I'm going to die",
        "Generalized anxiety disorder with excessive worry patterns",
        "Social anxiety preventing normal interactions with others",
        "Racing thoughts and inability to concentrate due to anxiety",
        "Physical symptoms include sweating, trembling, and nausea",
        "Avoidance behaviors due to anticipatory anxiety responses"
    ]
    
    suicide_samples = [
        "I have thoughts of ending my life and have made specific plans",
        "Patient expresses active suicidal ideation with clear intent and method",
        "Suicide attempt last week, still having active harmful thoughts",
        "Life feels meaningless, considering suicide as the only remaining option",
        "Previous multiple suicide attempts, current ideation with detailed plan",
        "Feeling like a burden to everyone, family would be better off without me",
        "Research methods of suicide online, have written goodbye letters to family",
        "Immediate safety concern due to active suicidal planning and intent",
        "Want to kill myself, have pills saved up for overdose",
        "Thinking about jumping off bridge, scouted locations already",
        "Cannot take this pain anymore, ready to end everything tonight",
        "Suicidal thoughts every day, plan to hang myself this weekend",
        "Told therapist about suicide plan, have rope and location picked",
        "Feel hopeless beyond repair, death seems like only escape from suffering",
        "Active suicidal ideation with intent, plan, and means available",
        "Patient admits to ongoing thoughts of self-harm and ending life permanently"
    ]
    
    # Generate balanced dataset - ensure we get exactly num_samples
    data = []
    samples_per_class = num_samples // 3
    
    # Create exactly samples_per_class of each type
    for _ in range(samples_per_class):
        data.append({'text': random.choice(depression_samples), 'label': 'depression'})
        data.append({'text': random.choice(anxiety_samples), 'label': 'anxiety'})
        data.append({'text': random.choice(suicide_samples), 'label': 'suicide'})
    
    # Add remaining samples to reach exact num_samples
    remaining = num_samples - len(data)
    all_samples = [
        (depression_samples, 'depression'),
        (anxiety_samples, 'anxiety'), 
        (suicide_samples, 'suicide')
    ]
    
    for i in range(remaining):
        sample_type = all_samples[i % 3]
        data.append({'text': random.choice(sample_type[0]), 'label': sample_type[1]})
    
    # Add some noise and variation
    for item in data:
        # Add some random medical terms
        if random.random() < 0.3:
            medical_terms = ["patient", "symptoms", "reported", "history of", "diagnosis", "treatment"]
            item['text'] = f"{random.choice(medical_terms)} {item['text']}"
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
    df.to_csv(output_path, index=False)
    
    logging.info(f"Sample dataset with {len(df)} samples saved to {output_path}")
