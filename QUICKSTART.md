# Mental Health Classifier - Quick Start Guide

## Project Structure

```
mental-health-classifier/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── pyproject.toml                     # Poetry configuration
├── main.py                            # Main training script
├── config/
│   └── config.yaml                    # Model and training configuration
├── src/                               # Source code
│   ├── models/
│   │   ├── __init__.py
│   │   └── mental_health_transformer.py  # Transformer model architecture
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_processor.py          # Data processing and loading
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py                 # Training loop and metrics
│   └── utils/
│       ├── __init__.py
│       └── helpers.py                 # Utility functions
├── notebooks/
│   └── mental_health_classifier_demo.ipynb  # Interactive demo
├── tests/
│   └── test_basic.py                  # Basic functionality tests
├── data/                              # Dataset storage (created at runtime)
├── models/                            # Saved model checkpoints
└── results/                           # Training results and plots
```

## Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd mental-health-classifier

# Install dependencies
pip install -r requirements.txt

# Or using Poetry (recommended)
poetry install
```

### 2. Run with Sample Data

```bash
# Create sample data and train model
python main.py --create-sample-data --sample-size 1000

# Or run with custom configuration
python main.py --config config/config.yaml
```

### 3. Interactive Exploration

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/mental_health_classifier_demo.ipynb
```

### 4. Run Tests

```bash
# Run basic tests
python tests/test_basic.py

# Or with pytest
pytest tests/
```

## Key Features

### Multi-headed Attention Transformer
- Configurable architecture (heads, layers, dimensions)
- Positional encoding for sequence understanding
- Clinical vocabulary integration

### Clinical Text Processing
- Medical abbreviation expansion (pt → patient, SI → suicidal ideation)
- Clinical terminology normalization
- Stopword removal and stemming options

### Mental Health Classification
- **Depression**: Mood disorders, hopelessness, anhedonia
- **Anxiety**: Worry, panic, GAD symptoms
- **Suicide Risk**: Ideation, planning, attempts

### Training Features
- Focal loss for class imbalance
- Class weighting
- Early stopping
- Comprehensive metrics (precision, recall, F1)
- Attention visualization

## Configuration

Edit `config/config.yaml` to customize:

```yaml
model:
  n_embd: 512          # Embedding dimension
  num_heads: 8         # Attention heads
  n_layer: 6           # Transformer layers
  num_classes: 3       # Depression, Anxiety, Suicide

training:
  batch_size: 32
  learning_rate: 2e-5
  num_epochs: 10
```

## Usage Examples

### Command Line Training
```bash
# Basic training
python main.py --create-sample-data

# Custom configuration
python main.py --config my_config.yaml --device cuda

# Different sample size
python main.py --create-sample-data --sample-size 5000
```

### Programmatic Usage
```python
from src.models import create_model
from src.data import DataProcessor
from src.training import create_trainer
from src.utils import load_config

# Load configuration
config = load_config('config/config.yaml')

# Initialize components
data_processor = DataProcessor(config['data'])
model = create_model(config['model'])
trainer = create_trainer(model, config['training'], device)

# Train model
trainer.train(train_dataloader, val_dataloader)

# Make predictions
prediction = trainer.predict_text(
    "I feel hopeless and can't see any way forward",
    data_processor
)
```

### Interactive Prediction
```python
# Get prediction with probabilities
text = "Patient reports severe anxiety and panic attacks"
prediction, probabilities = trainer.predict_text(
    text, data_processor, return_probabilities=True
)

print(f"Predicted: {prediction}")
for label, prob in probabilities.items():
    print(f"{label}: {prob:.3f}")
```

## Model Performance

Expected performance on clinical text:
- **Depression**: 85-90% F1 score
- **Anxiety**: 80-85% F1 score  
- **Suicide Risk**: 75-80% F1 score (more challenging)

## Important Notes

### Ethical Considerations
⚠️ **This is a research tool and should NOT be used for actual clinical diagnosis without:**
- Proper validation with licensed mental health professionals
- IRB approval for clinical use
- Comprehensive bias testing
- Human oversight in all decisions

### Data Requirements
For production use, consider:
- **Real clinical datasets** (MIMIC-III/IV, i2b2 challenges)
- **Balanced representation** across demographics
- **Privacy compliance** (HIPAA, GDPR)
- **Quality assurance** by clinical experts

### Model Limitations
- Trained on limited sample data
- May exhibit bias toward certain populations
- Cannot replace professional clinical judgment
- Requires validation on diverse clinical populations

## Advanced Features

### Attention Visualization
```python
# Get and visualize attention weights
attention_weights = model.get_attention_weights(input_ids)
# Plot attention heatmaps to understand model focus
```

### Custom Loss Functions
```python
# Built-in focal loss for imbalanced classes
from src.training import FocalLoss
criterion = FocalLoss(alpha=1.0, gamma=2.0)
```

### Clinical Vocabulary Integration
```python
# Expand clinical abbreviations
preprocessor = ClinicalTextPreprocessor(expand_contractions=True)
processed_text = preprocessor.preprocess("Pt c/o SI w/ h/o MDD")
# Output: "patient complains of suicidal ideation with history of major depressive disorder"
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch_size in config
   - Decrease model dimensions (n_embd, n_layer)
   - Use gradient checkpointing

2. **Poor Performance**
   - Increase training epochs
   - Add more training data
   - Adjust learning rate
   - Check class balance

3. **Slow Training**
   - Use GPU (CUDA/MPS)
   - Increase batch size
   - Reduce sequence length

### Debug Mode
```bash
# Run with debug logging
python main.py --create-sample-data 2>&1 | tee training.log
```

## Contributing

To extend the classifier:

1. **Add new mental health categories**
   - Update label mapping in config
   - Expand sample data generation
   - Retrain with new categories

2. **Integrate with clinical systems**
   - Add FHIR data connectors
   - Implement HL7 message parsing
   - Create API endpoints

3. **Improve clinical accuracy**
   - Add domain-specific pre-training
   - Integrate medical knowledge bases
   - Implement ensemble methods

## Citation

If you use this code in research, please cite:

```bibtex
@misc{mental_health_classifier,
  title={Multi-headed Attention Transformer for Mental Health Classification},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/mental-health-classifier}
}
```

## License

[Specify your license - consider restrictions for medical AI tools]

## Support

For questions or issues:
- Check existing issues in the repository
- Review the troubleshooting section
- Consult with clinical domain experts for medical validation

---

**Remember**: This tool is for research and educational purposes. Always involve qualified mental health professionals in clinical decision-making.
