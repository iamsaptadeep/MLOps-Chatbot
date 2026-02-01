
# MLOps Customer Support Chatbot

## Overview

This project implements an end-to-end MLOps pipeline for a production-style AI-powered customer support chatbot.

It demonstrates the complete lifecycle of a machine learning system, including data ingestion, training, experiment tracking, deployment, monitoring, and retraining.

The system performs intent classification on natural language queries and generates automated responses through a web-based interface.

The primary focus of this project is building a reliable, reproducible, and deployable machine learning system rather than a standalone model.

---

## Live Deployment

Frontend (Streamlit UI):
https://mlops-chatbot-ui.onrender.com

Backend (FastAPI API):
https://mlops-chatbot-api.onrender.com

Source Code:
https://github.com/iamsaptadeep/MLOps-Chatbot

---

## Key Features

- NLP-based intent classification using sentence embeddings
- Experiment tracking and artifact management with MLflow
- Dataset and model versioning using DVC and Git
- RESTful API built with FastAPI
- Web interface built with Streamlit
- Containerized deployment using Docker and Docker Compose
- Dependency version pinning for reproducibility
- Hybrid rule-based and ML-based inference
- Inference logging for monitoring
- Data drift detection using Evidently
- Manual retraining pipeline
- Cloud deployment with Docker-based services

---

## System Architecture

```

User
|
v
Streamlit Web UI
|
v
FastAPI Backend Service
|
v
Sentence Embedding Model
|
v
Intent Classifier
|
v
Prediction, Logging, and Monitoring
|
v
Retraining Pipeline

```

---

## Technology Stack

- Programming Language: Python
- Machine Learning: SentenceTransformers, Scikit-learn
- API Framework: FastAPI
- Frontend: Streamlit
- Experiment Tracking: MLflow
- Data Versioning: DVC
- Monitoring: Evidently
- Containerization: Docker, Docker Compose
- Version Control: Git
- Cloud Deployment: Render, Streamlit Cloud

---

## Dataset

The system is trained using the Bitext Customer Support Dataset.

This dataset contains synthetic customer support conversations with labeled intents and response templates.

Location:

```

data/raw/bitext_customer_support.csv

```

The dataset is tracked using DVC for version control and reproducibility.

---

## Project Structure

```

mlops-chatbot/
│
├── api/
│   └── main.py
│
├── ui/
│   └── app.py
│
├── training/
│   ├── train.py
│   └── retrain.py
│
├── monitoring/
│   ├── logger.py
│   ├── drift_report.py
│   └── data/
│
├── artifacts/
│   ├── classifier.pkl
│   └── encoder.pkl
│
├── data/
│   └── raw/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

````

---

## Training Pipeline

The training pipeline performs the following steps:

1. Load and preprocess the dataset
2. Generate sentence embeddings using SentenceTransformer
3. Train a Logistic Regression classifier
4. Evaluate model performance using accuracy and macro F1 score
5. Log metrics and parameters using MLflow
6. Store trained artifacts
7. Track artifacts using DVC

Run training:

```bash
python training/train.py
````

Run retraining:

```bash
python training/retrain.py
```

---

## Experiment Tracking

MLflow is used for tracking experiments and managing model artifacts.

It logs:

* Training metrics
* Hyperparameters
* Model versions
* Serialized artifacts

Start MLflow UI:

```bash
mlflow ui
```

Access:

```
http://localhost:5000
```

---

## API Service

The backend service exposes the trained model through a REST API.

Main endpoint:

```
POST /chat
```

Request format:

```json
{
  "message": "I want a refund"
}
```

Response format:

```json
{
  "intent": "get_refund",
  "response": "Your refund request has been received.",
  "confidence": 0.95
}
```

Swagger documentation:

```
/docs
```

---

## Web Interface

The Streamlit frontend provides an interactive chat interface.

Features:

* Real-time inference
* Session-based conversation history
* Confidence score display
* Error handling and fallback messaging

The UI communicates with the backend API via HTTP.

---

## Hybrid Inference Strategy

The inference system follows a hybrid approach:

1. Rule-based routing for high-priority keywords
2. Machine learning classification for general queries
3. Confidence thresholding for uncertainty handling
4. Fallback to human support when required

This improves robustness and reliability in production environments.

---

## Containerized Deployment

The application is containerized using Docker.

Docker Compose orchestrates:

* FastAPI backend service
* Streamlit frontend service

Build and run:

```bash
docker-compose up --build
```

Local access:

UI:

```
http://localhost:8501
```

API:

```
http://localhost:8000/docs
```

---

## Dependency Management

All dependencies are pinned in `requirements.txt`.

This ensures:

* Reproducible environments
* Compatibility with serialized models
* Stable production deployments

Torch CPU wheels are installed separately in Docker for Linux compatibility.

---

## Monitoring and Logging

### Inference Logging

Each API request is logged with:

* Timestamp
* User input
* Predicted intent
* Confidence score

Log location:

```
monitoring/data/inference_log.csv
```

---

### Drift Detection

Evidently is used to compare training data with production inference data.

Generate drift report:

```bash
python monitoring/drift_report.py
```

Output:

```
monitoring/reports/drift.html
```

---

## Retraining Pipeline

Retraining can be triggered when drift is detected or performance degrades.

The retraining workflow:

1. Reload dataset
2. Regenerate embeddings
3. Train new classifier
4. Log results to MLflow
5. Update model artifacts
6. Version new artifacts using DVC

Command:

```bash
python training/retrain.py
```

---

## Version Control and Reproducibility

This project uses:

* Git for source control
* DVC for dataset and model tracking
* MLflow for experiment management

This enables full reproducibility of training and deployment pipelines.

Any historical version of the system can be restored.

---

## Local Development (Without Docker)

1. Create virtual environment

```bash
python -m venv venv
```

2. Activate environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Start API

```bash
uvicorn api.main:app --reload
```

5. Start UI

```bash
streamlit run ui/app.py
```

---

## Future Improvements

* Automated retraining using schedulers
* CI/CD pipeline with GitHub Actions
* Online learning support
* Model calibration
* Cloud storage integration
* User feedback-driven learning
* LLM-based fallback responses

---

## Author

Developed as a hands-on MLOps portfolio project focused on building production-grade machine learning systems.

````

---
