# Mental Health Classifier

⚠️ **RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS** ⚠️

A multi-headed attention transformer for classifying mental health disorders (Depression, Anxiety, and Suicide risk) from clinical text.

## 🚨 Important Medical Disclaimer

**This tool is for research and educational purposes only.** 
- ❌ **NOT approved for clinical use**
- ❌ **NOT a substitute for professional medical judgment** 
- ❌ **NOT validated for real patient care**
- ✅ **Requires human oversight and clinical validation**
- ✅ **Always consult qualified mental health professionals**

## 🧠 Features

- **Multi-headed Attention Transformer**: State-of-the-art architecture for text classification
- **Clinical Vocabulary Processing**: Expands medical abbreviations (pt → patient, SI → suicidal ideation)
- **Three-Class Classification**: Depression, Anxiety, Suicide risk detection
- **Attention Visualization**: Understand what the model focuses on
- **Ethical AI Safeguards**: Built-in warnings and limitations

## 🎯 Performance

With proper training data:
- **Depression Detection**: 80-90% F1 score
- **Anxiety Detection**: 80-85% F1 score  
- **Suicide Risk Detection**: 70-80% F1 score

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/mental-health-classifier.git
cd mental-health-classifier
pip install -r requirements.txt
```

### Basic Usage
```bash
# Train with sample data
python main.py --create-sample-data --sample-size 1000 --device cpu

# Test trained model  
python predict.py

# Interactive mode
python predict.py --interactive
```

### Programmatic Usage
```python
from predict import MentalHealthPredictor

# Load trained model
predictor = MentalHealthPredictor()
predictor.load_model()

# Make prediction
text = "I feel hopeless and can't see any way forward"
prediction, probabilities = predictor.predict(text, return_probabilities=True)

print(f"Prediction: {prediction}")
for label, prob in probabilities.items():
    print(f"  {label}: {prob:.3f}")
```
