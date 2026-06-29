# 🩻 RadiologyAI — Automatic Radiology Report Generation

> **Deep Learning · Vision-Language Models · EfficientNet-B3 + Transformer**

An end-to-end AI system that takes a chest X-ray image as input and automatically
generates a professional radiology report with abnormality detection and
Grad-CAM explainability visualisations.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Project Structure](#-project-structure)
4. [Quick Start](#-quick-start)
5. [Dataset Setup](#-dataset-setup)
6. [Training](#-training)
7. [Inference & API](#-inference--api)
8. [Frontend UI](#-frontend-ui)
9. [Evaluation](#-evaluation)
10. [Docker Deployment](#-docker-deployment)
11. [Streamlit Demo](#-streamlit-demo)
12. [Testing](#-testing)
13. [Configuration Reference](#-configuration-reference)
14. [Results](#-results)
15. [Roadmap](#-roadmap)

---

## 🎯 Project Overview

RadiologyAI automates the generation of radiology reports from chest X-ray images using
state-of-the-art deep learning:

| Feature                    | Detail                                              |
|---------------------------|-----------------------------------------------------|
| **Task**                  | Image-to-text (Image Captioning / VLM)             |
| **Visual Encoder**        | EfficientNet-B3 (ImageNet pre-trained)             |
| **Language Decoder**      | 6-layer Transformer with Pre-LN                    |
| **Decoding**              | Beam Search (width 5)                              |
| **Explainability**        | Grad-CAM overlays + Transformer attention maps     |
| **Evaluation**            | BLEU-1/2/3/4, ROUGE-1/2/L, METEOR, token accuracy |
| **Datasets**              | OpenI Indiana University CXR, MIMIC-CXR            |
| **Backend**               | FastAPI + Python                                   |
| **Frontend**              | Vanilla HTML/CSS/JS (zero-dependency)              |
| **Deployment**            | Docker + docker-compose                            |

---

## 🏗 Architecture

```
Chest X-Ray Image (224×224×3)
        │
        ▼
┌──────────────────────┐
│  EfficientNet-B3     │  Pre-trained visual encoder
│  Feature Extractor   │  Outputs: (B, S, 512) spatial tokens
└──────────┬───────────┘
           │  cross-attention
           ▼
┌──────────────────────┐
│  Transformer Decoder │  6 layers, 8 heads, Pre-LN
│  (Autoregressive)    │  Teacher-forced training
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Vocabulary Softmax  │  Label smoothing (ε=0.1)
│  + Beam Search       │  Beam width = 5
└──────────┬───────────┘
           │
           ▼
  Generated Radiology Report (text)
```

### Key Design Choices

- **Pre-Layer Normalisation** — more stable training for deep decoders
- **Weight Tying** — embedding ↔ output projection (reduces parameters ~30%)
- **Warmup + Cosine Decay** — smooth LR schedule prevents early overfitting
- **Mixed Precision (FP16)** — 2× memory efficiency, faster training on GPU
- **Label Smoothing** — reduces overconfidence, improves generalisation

---

## 📁 Project Structure

```
radiology-ai/
├── config/
│   └── config.yaml              # Master configuration
│
├── ai_model/
│   ├── models/
│   │   └── radiology_model.py   # EfficientNet-B3 + Transformer
│   ├── training/
│   │   └── trainer.py           # Full training loop
│   ├── evaluation/
│   │   └── metrics.py           # BLEU, ROUGE, METEOR
│   ├── explainability/
│   │   └── gradcam.py           # Grad-CAM + attention maps
│   └── inference.py             # Inference engine (singleton)
│
├── datasets/
│   ├── scripts/
│   │   ├── prepare_dataset.py   # Download, preprocess, tokenise
│   │   └── dataloader.py        # PyTorch Dataset + DataLoader
│   ├── raw/                     # Downloaded raw data (gitignored)
│   └── processed/               # Preprocessed images + annotations
│
├── backend/
│   ├── main.py                  # FastAPI app
│   ├── api/
│   │   └── schemas.py           # Pydantic request/response models
│   ├── services/
│   │   └── inference_service.py # Model singleton accessor
│   └── utils/
│       └── image_utils.py       # PIL + DICOM helpers
│
├── frontend/
│   └── index.html               # Dark modern UI (zero dependencies)
│
├── docker/
│   ├── Dockerfile               # Multi-stage backend image
│   ├── docker-compose.yml       # Full stack orchestration
│   └── nginx.conf               # Frontend reverse proxy
│
├── notebooks/
│   └── exploration.ipynb        # Training curves, Grad-CAM, metrics
│
├── tests/
│   └── test_suite.py            # Comprehensive pytest suite
│
├── streamlit_demo.py            # Quick Streamlit demo
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional but recommended)
- 8 GB+ RAM, 16 GB+ VRAM for training

### 1. Clone & Install

```bash
git clone https://github.com/your-username/radiology-ai.git
cd radiology-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for METEOR metric)
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 2. Prepare Dataset

```bash
# OpenI dataset (recommended for quick start)
python datasets/scripts/prepare_dataset.py \
    --dataset openI \
    --data_dir datasets/raw/ \
    --out_dir  datasets/processed/ \
    --download               # requires Kaggle API key configured
```

> **Manual download**: If you don't have Kaggle configured, download from
> https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university
> and extract to `datasets/raw/`.

### 3. Train

```bash
python ai_model/training/trainer.py --config config/config.yaml
```

### 4. Start API

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open Frontend

```bash
# Simply open in browser
open frontend/index.html
# Or serve with Python
python -m http.server 3000 --directory frontend
```

---

## 📊 Dataset Setup

### OpenI Indiana University (Recommended)

The OpenI dataset is freely available and contains:
- ~7,000 frontal/lateral chest X-ray images
- XML reports with FINDINGS and IMPRESSION sections
- No access restrictions

```bash
# With Kaggle CLI (configure ~/.kaggle/kaggle.json first)
kaggle datasets download -d raddar/chest-xrays-indiana-university \
    --unzip -p datasets/raw/
```

### MIMIC-CXR

The MIMIC-CXR dataset contains 227,827 imaging studies but requires:
1. PhysioNet account: https://physionet.org
2. Completion of CITI training course
3. Data use agreement

```bash
# After access granted:
wget -r -N -c -np \
    --user YOUR_USERNAME --password YOUR_PASSWORD \
    https://physionet.org/files/mimic-cxr/2.0.0/
```

### Preprocessing Pipeline

```
Raw XML + PNG
     │
     ├─ Parse XML reports → extract FINDINGS + IMPRESSION
     ├─ Clean text (normalize whitespace, handle anonymisation)
     ├─ Resize images to 224×224 (LANCZOS)
     ├─ Build vocabulary (min_freq=3)
     ├─ Tokenise reports (word-level, max_len=150)
     └─ Split 80/10/10 → annotations.json
```

---

## 🏋 Training

### Configuration (`config/config.yaml`)

```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 1.0e-4
  mixed_precision: true      # FP16 on GPU
  early_stopping_patience: 7
```

### Training Commands

```bash
# Basic training
python ai_model/training/trainer.py

# Custom config
python ai_model/training/trainer.py --config config/config.yaml

# Resume from checkpoint
python ai_model/training/trainer.py --resume ai_model/checkpoints/checkpoint_epoch_020.pth
```

### Monitor with TensorBoard

```bash
tensorboard --logdir logs/tensorboard --port 6006
# Open http://localhost:6006
```

### Training Tips

| Scenario          | Recommendation                                       |
|-------------------|------------------------------------------------------|
| **Limited GPU**   | Reduce `batch_size` to 8–16, enable `mixed_precision`|
| **Slow progress** | Increase `warmup_steps` to 1000                      |
| **Overfitting**   | Increase `dropout` to 0.2, reduce `batch_size`       |
| **MIMIC-CXR**     | Use `batch_size=64`, `epochs=30`                     |
| **CPU only**      | Set `batch_size=4`, `num_workers=0`                  |

---

## 🔌 Inference & API

### REST API Endpoints

| Method | Endpoint        | Description                                      |
|--------|-----------------|--------------------------------------------------|
| `GET`  | `/health`       | Liveness check                                   |
| `POST` | `/predict`      | Upload X-ray → generated report + Grad-CAM       |
| `GET`  | `/evaluate`     | Run evaluation on test set                       |
| `POST` | `/train`        | Trigger background training run                  |
| `GET`  | `/train/status` | Training job status                              |

### Predict Endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@chest_xray.png" \
  -F "beam_size=5" \
  -F "gradcam=true"
```

**Response:**
```json
{
  "uid": "a3f2e1b4c5d6",
  "report": "The lungs are clear bilaterally. No pleural effusion or pneumothorax identified. Cardiomediastinal silhouette within normal limits.",
  "confidence": 0.87,
  "gradcam_url": "/static/gradcam/a3f2e1b4c5d6_gradcam.png",
  "latency_ms": 342.5
}
```

### API Docs

Interactive Swagger UI available at: `http://localhost:8000/docs`

---

## 🖥 Frontend UI

The frontend is a single `index.html` with zero npm dependencies:

- **Drag-and-drop** image upload
- **Live preview** with original / Grad-CAM tab switching
- **Animated report** display with keyword highlighting
- **Confidence meter** with colour coding
- **Key findings** extraction (abnormality detection)
- **Copy to clipboard** and **API health indicator**
- **Dark mode** with ambient gradients

```bash
# Serve locally
python -m http.server 3000 --directory frontend
# Navigate to http://localhost:3000
```

---

## 📈 Evaluation

### Run Evaluation

```bash
# Via API
curl http://localhost:8000/evaluate

# Via Python
python -c "
from ai_model.inference import RadiologyInferenceEngine
engine  = RadiologyInferenceEngine.from_config('config/config.yaml')
metrics = engine.evaluate_on_test_set()
for k, v in metrics.items():
    print(f'{k:20s} {v:.4f}')
"
```

### Metrics

| Metric          | Description                                             |
|-----------------|---------------------------------------------------------|
| **BLEU-1/2/3/4**| N-gram precision vs reference (modified)               |
| **ROUGE-1/2/L** | Recall-oriented overlap with reference                  |
| **METEOR**      | Considers synonyms, stemming                            |
| **Token Acc.**  | Per-token prediction accuracy                           |

---

## 🐳 Docker Deployment

```bash
# Build and start all services
cd docker
docker-compose up --build -d

# Check logs
docker-compose logs -f backend

# Stop
docker-compose down
```

Services:
- **Backend API** → http://localhost:8000
- **Frontend UI** → http://localhost:3000
- **API Docs**    → http://localhost:8000/docs

### GPU Support

Enable NVIDIA GPU in `docker-compose.yml` (already configured):
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

---

## ✨ Streamlit Demo

```bash
pip install streamlit
streamlit run streamlit_demo.py
# Opens at http://localhost:8501
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/test_suite.py -v

# Run specific test classes
pytest tests/test_suite.py::TestRadiologyModel -v
pytest tests/test_suite.py::TestMetrics -v
pytest tests/test_suite.py::TestAPI -v

# With coverage report
pip install pytest-cov
pytest tests/test_suite.py --cov=. --cov-report=html
```

---

## ⚙️ Configuration Reference

All settings live in `config/config.yaml`:

```yaml
dataset:
  image_size: 224         # Input image size (px)
  max_seq_len: 150        # Max report tokens
  min_freq: 3             # Vocabulary min word frequency

model:
  encoder: "efficientnet_b3"
  d_model: 512
  num_heads: 8
  num_decoder_layers: 6

training:
  epochs: 50
  batch_size: 32
  learning_rate: 1.0e-4
  mixed_precision: true
  early_stopping_patience: 7

inference:
  beam_size: 5
  max_len: 150
  min_len: 20
```

---

## 📊 Results

Expected performance on OpenI test set (after 50 epochs):

| Metric   | Score  |
|----------|--------|
| BLEU-1   | ~0.42  |
| BLEU-4   | ~0.18  |
| ROUGE-1  | ~0.46  |
| ROUGE-L  | ~0.38  |
| METEOR   | ~0.31  |

*Scores are approximate and depend on dataset size, training duration, and hardware.*

---

## 🗺 Roadmap

- [ ] BioGPT / LLaVA fine-tuning integration
- [ ] MIMIC-CXR full pipeline automation
- [ ] ClinicalBERT semantic similarity metric
- [ ] Multi-view (frontal + lateral) fusion
- [ ] Structured report output (sections: Findings, Impression)
- [ ] Hugging Face Hub model upload
- [ ] ONNX export for edge deployment
- [ ] DICOM viewer integration

---

## 📜 License

MIT License — see `LICENSE` for details.

## ⚠️ Disclaimer

**This system is for research and educational purposes only.**
It must NOT be used for clinical diagnosis or medical decision-making
without validation by qualified radiologists.

---

*Built with ❤️ using PyTorch, FastAPI, and EfficientNet-B3.*
