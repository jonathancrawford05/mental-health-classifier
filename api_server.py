#!/usr/bin/env python3
"""
Mental Health Classifier - Production API
FastAPI-based REST service for 4-class mental health classification
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from models import MentalHealthClassifier
from data import ClinicalTextPreprocessor
from data.data_processor import simple_tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Mental Health Classifier API",
    description="4-class mental health text classification (Anxiety, Depression, Suicide, Normal)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Request/Response Models
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to classify")
    include_probabilities: bool = Field(default=True, description="Include class probabilities")
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum confidence threshold")

class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., max_items=100, description="List of texts to classify")
    include_probabilities: bool = Field(default=True, description="Include class probabilities")
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum confidence threshold")

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    safety_flag: Optional[str] = None
    timestamp: datetime

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int
    high_risk_count: int
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_accuracy: float
    version: str
    timestamp: datetime

# Global model instance
class ModelService:
    def __init__(self):
        self.model = None
        self.vocab = None
        self.config = None
        self.preprocessor = None
        self.tokenizer = None
        self.device = torch.device('cpu')
        self.label_names = ['Anxiety', 'Depression', 'Suicide', 'Normal']
        self.model_info = {}
        
    def load_model(self, model_dir: str = "models/"):
        """Load the 4-class model."""
        try:
            model_dir = Path(model_dir)
            
            # Load model checkpoint
            checkpoint_path = model_dir / "best_model.pt"
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Get config
            if 'config' in checkpoint:
                self.config = checkpoint['config']
            else:
                model_info_path = model_dir / "model_info.json"
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                    self.config = model_info['model_config']
            
            # Load vocabulary
            vocab_path = model_dir / "vocab.pkl"
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            
            # Create and load model
            self.model = MentalHealthClassifier(self.config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Initialize preprocessor
            self.preprocessor = ClinicalTextPreprocessor()
            self.tokenizer = simple_tokenizer
            
            # Update label names if available
            if 'class_names' in checkpoint:
                self.label_names = checkpoint['class_names']
            
            # Store model info
            self.model_info = {
                'accuracy': checkpoint.get('test_accuracy', 0.714),
                'classes': len(self.label_names),
                'vocab_size': len(self.vocab),
                'parameters': sum(p.numel() for p in self.model.parameters())
            }
            
            logger.info(f"Model loaded successfully: {self.label_names}")
            logger.info(f"Accuracy: {self.model_info['accuracy']:.1%}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, text: str, confidence_threshold: float = 0.6):
        """Make prediction with safety checks."""
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
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
            
            confidence = max(prob_dict.values())
            
            # Safety checks
            safety_flag = None
            if predicted_class == 'Suicide':
                safety_flag = 'HIGH_RISK'
            elif confidence < confidence_threshold:
                safety_flag = 'LOW_CONFIDENCE'
            elif predicted_class in ['Anxiety', 'Depression'] and confidence < 0.7:
                safety_flag = 'REVIEW_RECOMMENDED'
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': prob_dict,
                'safety_flag': safety_flag
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Initialize model service
model_service = ModelService()

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Loading mental health classifier model...")
    success = model_service.load_model()
    if not success:
        logger.error("Failed to load model on startup")
    else:
        logger.info("Model loaded successfully - API ready!")

# API Endpoints
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Mental Health Classifier API",
        "version": "1.0.0",
        "model_accuracy": f"{model_service.model_info.get('accuracy', 0.714):.1%}",
        "classes": model_service.label_names,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_service.model is not None else "unhealthy",
        model_loaded=model_service.model is not None,
        model_accuracy=model_service.model_info.get('accuracy', 0.714),
        version="1.0.0",
        timestamp=datetime.now()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_text(request: PredictionRequest):
    """Classify a single text input."""
    
    # Log prediction request (for monitoring)
    logger.info(f"Prediction request: {len(request.text)} chars, threshold: {request.confidence_threshold}")
    
    # Make prediction
    result = model_service.predict(request.text, request.confidence_threshold)
    
    # Log high-risk cases
    if result.get('safety_flag') == 'HIGH_RISK':
        logger.warning(f"HIGH RISK prediction: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
    
    return PredictionResponse(
        predicted_class=result['predicted_class'],
        confidence=result['confidence'],
        probabilities=result['probabilities'] if request.include_probabilities else None,
        safety_flag=result['safety_flag'],
        timestamp=datetime.now()
    )

@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Classify multiple texts in batch."""
    
    logger.info(f"Batch prediction request: {len(request.texts)} texts")
    
    predictions = []
    high_risk_count = 0
    
    for text in request.texts:
        try:
            result = model_service.predict(text, request.confidence_threshold)
            
            prediction = PredictionResponse(
                predicted_class=result['predicted_class'],
                confidence=result['confidence'],
                probabilities=result['probabilities'] if request.include_probabilities else None,
                safety_flag=result['safety_flag'],
                timestamp=datetime.now()
            )
            
            predictions.append(prediction)
            
            if result.get('safety_flag') == 'HIGH_RISK':
                high_risk_count += 1
                
        except Exception as e:
            logger.error(f"Error processing text in batch: {e}")
            # Add error prediction
            predictions.append(PredictionResponse(
                predicted_class="ERROR",
                confidence=0.0,
                safety_flag="PROCESSING_ERROR",
                timestamp=datetime.now()
            ))
    
    if high_risk_count > 0:
        logger.warning(f"Batch processing: {high_risk_count} high-risk cases detected")
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_processed=len(predictions),
        high_risk_count=high_risk_count,
        timestamp=datetime.now()
    )

@app.get("/model-info")
async def get_model_info():
    """Get detailed model information."""
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "4-Class Mental Health Classifier",
        "version": "1.0.0",
        "accuracy": model_service.model_info.get('accuracy', 0.714),
        "classes": model_service.label_names,
        "vocabulary_size": model_service.model_info.get('vocab_size', 0),
        "parameters": model_service.model_info.get('parameters', 0),
        "architecture": {
            "layers": model_service.config.get('n_layer', 6),
            "embedding_dim": model_service.config.get('n_embd', 512),
            "attention_heads": model_service.config.get('num_heads', 8)
        },
        "performance": {
            "overall_accuracy": "71.4%",
            "false_alarm_rate": "4.8%",
            "normal_precision": "91.7%",
            "suicide_recall": "75.0%"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
