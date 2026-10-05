# Named Entity Recognition System

> Precision medical entity detection for clinical text using transformer-based NLP.

Repository: https://github.com/sakthiiiiiiii/Named_Entity_Recogition_System

## Overview
This repository contains a Google Colab-ready workflow for fine-tuning a Named Entity Recognition (NER) model for medical and clinical text. The project uses spaCy’s transformer pipeline with a biomedical language model to detect entities such as diseases, medications, symptoms, procedures, and other clinically relevant terms.

The notebook in this repository demonstrates how to set up the environment, verify GPU support, mount Google Drive, prepare training data, generate a spaCy configuration, and train a transformer-based NER model for healthcare text.

## Key Features
- Fine-tunes a medical NER model using spaCy and transformers
- Uses BioClinicalBERT for biomedical text understanding
- Optimized for Google Colab and GPU-enabled training
- Supports training, validation, and evaluation workflows
- Stores datasets and model outputs in Google Drive
- Suitable for clinical text analysis and healthcare NLP tasks

## Tech Stack
- Python
- spaCy
- spaCy Transformers
- Hugging Face Transformers
- PyTorch
- Google Colab
- CUDA / GPU Training
- BioClinicalBERT

## Repository Structure
- `Medical_NER_Fine_Tuning_Colab.ipynb` — main notebook for model setup and training

## Installation
### Option 1: Google Colab
1. Open the notebook in Google Colab.
2. Run the cells sequentially from top to bottom.
3. Install dependencies:

```bash
!pip install -U spacy==3.8.14 spacy-transformers==1.3.9 transformers accelerate
```

4. Validate the installation:

```bash
!python -m spacy validate
```

5. Mount Google Drive if you are storing data or model files there:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Usage
### 1. Set dataset paths
```python
TRAIN_PATH = '/content/train.spacy'
DEV_PATH = '/content/dev.spacy'
TEST_PATH = '/content/test.spacy'
```

### 2. Initialize spaCy transformer config
```bash
!python -m spacy init config config.cfg \
  --lang en \
  --pipeline transformer,ner \
  --optimize accuracy \
  --gpu
```

### 3. Train the model
```bash
!python -m spacy train config.cfg \
  --output /content/drive/MyDrive/output.model \
  --paths.train /content/drive/MyDrive/train.spacy \
  --paths.dev /content/drive/MyDrive/dev.spacy \
  --gpu-id 0
```

### 4. Load the model
```python
import spacy

nlp = spacy.blank("en")

nlp.add_pipe(
    "transformer",
    config={
        "model": {
            "@architectures": "spacy-transformers.TransformerModel.v3",
            "name": "emilyalsentzer/Bio_ClinicalBERT",
        }
    },
)

print("Transformer loaded successfully!")
```

### 5. Run entity extraction
```python
import spacy

nlp = spacy.load("/path/to/trained/model")
text = "The patient was diagnosed with pneumonia and prescribed amoxicillin."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
```

## Example Use Cases
- Clinical note analysis
- Disease and symptom recognition
- Medication extraction
- Biomedical NLP research
- Healthcare text understanding

## Notes
- GPU support is recommended for faster and more efficient training.
- This project is tailored for medical-domain NER and works best with annotated clinical datasets.
- The workflow is easy to adapt for custom datasets and research experiments.

## Acknowledgments
This project builds on the capabilities of spaCy, Hugging Face Transformers, PyTorch, and the BioClinicalBERT model for biomedical entity recognition.
