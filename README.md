---
title: Canine Vision Classifier
emoji: 🐕
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.42.1"
app_file: app.py
pinned: false
---


# Canine Breed Classification

This repository contains a machine learning system for classifying 120 canine breeds from images using deep learning. The core is implemented with TensorFlow/Keras, deployed with Streamlit, containerized with Docker, and managed with MLflow.


This project demonstrates a full ML pipeline from data exploration to production deployment:

- **Training**: Transfer learning with ResNet50, EfficientNetB0, and fusion architectures
- **MLOps**: Model versioning and tracking with MLflow
- **Deployment**: Interactive web app with Streamlit
- **CI/CD**: Automated testing and Docker builds with GitHub Actions

**Dataset**: Stanford Dogs (20,580 images, 120 breeds)  
**Best Model**: EfficientNetB0 Fine-Tuned (~85% validation accuracy)

## Live Demo

**Try it now**: [Canine Vision Classifier on Hugging Face](https://huggingface.co/spaces/Roufa-HF/canine-vision-classifier)
   

Deployed automatically from GitHub using Hugging Face Spaces.

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/roufa-cloud/canine-vision-mlops.git
cd canine-vision-mlops
```

### 2. Install Dependencies

```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate  
# Windows PowerShell
venv\Scripts\Activate
pip install -r requirements.txt
```

### 3. Access the Dataset
   The Stanford Dogs dataset is downloaded automatically by the notebooks via `tensorflow_datasets`. No manual download scripts are required.

## Model Training

### Option 1: Use Notebooks

```bash
jupyter notebook
```

Open and run in order:
1. `00_eda.ipynb` - Data exploration
2. `02_resnet50.ipynb` - ResNet50 training
3. `03_efficientnetb0.ipynb` - EfficientNetB0 training
4. `04_feature_fusion.ipynb` - Fusion model
5. `05_evaluation_and_comparison.ipynb` - Compare models

### Option 2: Command Line

```bash
# Prepare artifacts
python scripts/prepare_deployment_artifacts.py

# Register in MLflow (optional)
python scripts/register_model_mlflow.py
```

## Deployment

### Local (Streamlit)

```bash
streamlit run app.py
```

### Docker

```bash
# Build and run
docker-compose up --build

# With MLflow UI
docker-compose --profile mlflow up
```

Access:
- **App**: http://localhost:8501
- **MLflow UI**: http://localhost:5000


## Testing

```bash
# Run all tests
python tests/test_inference.py

# Test system end-to-end
python scripts/test_inference.py --image assets/sample_images/dog.jpg
```

## Making Predictions

- **Via Streamlit app**: upload an image or select a sample; results are shown instantly.
- **Via script**: run `python scripts/test_inference.py` or with the `--image` option. The script prints the top prediction and confidence bars for the top‑k breeds.


## Model Performance

| Model | Val Accuracy | Top-5 Accuracy | Parameters |
|-------|--------------|----------------|------------|
| ResNet50 Fine-Tuned | 79.2% | 97.0% | 23.6M |
| EfficientNetB0 Fine-Tuned | 85.1% | 98.7% | 4.2M |
| Fusion (ResNet50 + EfficientNetB0) | 84.1% | 98.2% | 27.6M |

*Best single model: **EfficientNetB0** (best accuracy-to-size ratio)*

## Project Structure

```
dog-breed-classification/
├── notebooks/              # Training & evaluation notebooks
│   ├── 00_eda.ipynb
│   ├── 02_resnet50.ipynb
│   ├── 03_efficientnetb0.ipynb
│   ├── 04_feature_fusion.ipynb
│   └── 05_evaluation_and_comparison.ipynb
├── src/
│   ├── data/              # Dataset loading & preprocessing
│   ├── models/            # Model architectures
│   ├── training/          # Training utilities
│   ├── evaluation/        # Metrics & visualization
│   └── inference/         # Prediction engine
├── scripts/               # Utility scripts
│   ├── prepare_deployment_artifacts.py
│   └── register_model_mlflow.py
├── assets/                # Sample images, diagrams, logo
├── demo_images/           # Demo images of running app locally,  hugging-face and AWS EC2
├── config/                # YAML configuration files
├── tests/                 # Unit tests
├── artefacts/             # Models, metrics, MLflow data
├── requirements.txt       # Python dependencies
├── app.py                 # Streamlit web app
├── Dockerfile             # Container configuration
└── docker-compose.yml     # Multi-service setup
```

## Technology Stack

- ML Framework: TensorFlow, Keras
- UI: Streamlit
- Data: tensorflow_datasets (Stanford Dogs)
- Model tracking (optional): MLflow
- Deployment tooling: Docker
- CI/CD: GitHub Actions
- Testing: pytest



## Features

### Web App
- Upload or use sample images
- Real-time predictions
- Top-5 breed predictions with confidence
- Probability distribution for all 120 breeds
- Model metadata display

### MLflow Integration
- Experiment tracking
- Model versioning
- Automatic fallback to checkpoints
- Parameter and metric logging

### CI/CD Pipeline
- Automated testing on push
- Docker image builds
- GitHub Actions workflow
- Deployment-ready checks



## Development

### Add New Model

1. Create model in `src/models/your_model.py`
2. Add training notebook
3. Update config in `config/default.yaml`
4. Register with MLflow

### Retrain Models

```bash
# Run training notebooks
jupyter notebook notebooks/

# Prepare artifacts
python scripts/prepare_deployment_artifacts.py

# Register best model
python scripts/register_model_mlflow.py
```

### Run Tests Locally

```bash
# Install dev dependencies
pip install pytest

# Run tests
python tests/test_inference.py
```


## Demo

Check `assets/demo_images/` for screenshots of:
- Training notebooks
- MLflow UI
- Streamlit predictions
- Docker deployment




## License

MIT License - see [LICENSE](LICENSE) for details


