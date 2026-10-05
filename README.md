# Named Entity Recognition System

> Precision medical entity detection for clinical text using transformer-based NLP.

## Overview
This repository contains a Google Colab-ready workflow for fine-tuning a Named Entity Recognition (NER) model for medical and clinical text. The project uses spaCy's transformer pipeline with a biomedical language model, enabling high-quality extraction of entities such as diseases, medications, treatments, symptoms, and other clinically relevant terms.

The notebook in this repository demonstrates how to set up the environment, verify GPU availability, mount Google Drive, prepare training data, generate a spaCy configuration, and train a transformer-based NER model for medical text.

## Key Features
- Fine-tunes a medical NER model using spaCy and transformer architectures
- Uses BioClinicalBERT for biomedical domain adaptation
- Built for Google Colab with GPU acceleration support
- Supports training and validation splits for spaCy data files
- Integrates easily with Google Drive for dataset and model persistence
- Ready for downstream clinical NLP tasks and entity extraction workflows

## Tech Stack
- Python
- spaCy
- spaCy Transformers
- Hugging Face Transformers
- PyTorch
- Google Colab
- GPU-enabled training (CUDA)
- BioClinicalBERT

## Repository Structure
- `Medical_NER_Fine_Tuning_Colab.ipynb` — main training notebook for the medical NER workflow

## Installation
### Option 1: Google Colab (Recommended)
1. Open the notebook in Google Colab.
2. Run the notebook cells in order from top to bottom.
3. Install the required dependencies:

```bash
!pip install -U spacy==3.8.14 spacy-transformers==1.3.9 transformers accelerate
```

4. Validate the spaCy installation:

```bash
!python -m spacy validate
```

5. Mount your Google Drive if storing datasets or model outputs there:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Usage
### 1. Configure paths
Set your training, development, and test data paths:

```python
TRAIN_PATH = '/content/train.spacy'
DEV_PATH = '/content/dev.spacy'
TEST_PATH = '/content/test.spacy'
```

### 2. Generate a spaCy transformer config

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

### 4. Load the transformer model
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

### 5. Run inference
After training, load the model and pass clinical text through it to extract named entities:

```python
import spacy

nlp = spacy.load("/path/to/trained/model")
text = "The patient was diagnosed with pneumonia and prescribed amoxicillin."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
```

## Example Use Cases
- Clinical note entity extraction
- Medical record analysis
- Disease and treatment recognition
- Biomedical NLP research and prototyping
- Healthcare text intelligence workflows

## Notes
- GPU support is strongly recommended for faster training and improved experimentation speed.
- This project is designed for medical domain NER tasks and works best with domain-specific annotated datasets.
- The notebook is intended as a training and prototyping workflow, making it easy to adapt for custom datasets.

## License
This project is provided as-is for educational and research purposes. Please review the repository license if one is added later.

## Acknowledgments
This workflow relies on spaCy, Hugging Face Transformers, and the BioClinicalBERT biomedical language model for clinical text understanding.
