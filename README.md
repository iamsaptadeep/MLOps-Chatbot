
# MLOps Customer Support Chatbot

## Overview

This project implements an end-to-end MLOps pipeline for a customer support chatbot. It covers the complete lifecycle of a machine learning system, from data ingestion and training to deployment, monitoring, and retraining.

The system predicts user intent from natural language queries and returns automated support responses through a web-based interface.

The project focuses on building a production-style ML system rather than a standalone model.

---

## Key Features

* NLP-based intent classification using sentence embeddings
* Experiment tracking with MLflow
* Data and model versioning using DVC and Git
* REST API using FastAPI
* Web interface using Streamlit
* Containerized deployment with Docker and Docker Compose
* Dependency version locking for reproducibility
* Hybrid inference using rules and machine learning
* Inference logging for monitoring
* Data drift detection using Evidently
* Manual retraining pipeline

---

## System Architecture

```
User
  |
  v
Streamlit UI
  |
  v
FastAPI API
  |
  v
Embedding Model + Classifier
  |
  v
Prediction and Logging
  |
  v
Monitoring and Retraining
```

---

## Tech Stack

* Programming Language: Python
* Machine Learning: SentenceTransformers, Scikit-learn
* API Framework: FastAPI
* Frontend: Streamlit
* Experiment Tracking: MLflow
* Data Versioning: DVC
* Monitoring: Evidently
* Containerization: Docker, Docker Compose
* Version Control: Git

---

## Dataset

The project uses the Bitext Customer Support Dataset for intent classification.

The dataset contains synthetic customer queries and labeled intents with response templates.

It is stored under:

```
data/raw/bitext_customer_support.csv
```

and tracked using DVC.

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
```

---

## Training Pipeline

The training pipeline performs the following steps:

1. Load and preprocess dataset
2. Generate sentence embeddings using SentenceTransformer
3. Train a Logistic Regression classifier
4. Evaluate using accuracy and macro F1 score
5. Log metrics and artifacts using MLflow
6. Save trained models to the artifacts directory
7. Track artifacts using DVC

Training can be executed using:

```bash
python training/train.py
```

Retraining can be executed using:

```bash
python training/retrain.py
```

---

## Experiment Tracking

MLflow is used to track experiments and model performance.

It stores:

* Training metrics
* Hyperparameters
* Model artifacts

To start MLflow UI:

```bash
mlflow ui
```

Then open:

```
http://localhost:5000
```

---

## API Service

The FastAPI backend exposes the trained model as a REST service.

Main endpoint:

```
POST /chat
```

Example request:

```json
{
  "message": "I want a refund"
}
```

Example response:

```json
{
  "intent": "get_refund",
  "response": "Your refund request has been received.",
  "confidence": 0.95
}
```

Swagger documentation is available at:

```
/docs
```

---

## Web Interface

The Streamlit application provides a chat interface for end users.

It communicates with the FastAPI backend and displays predictions in real time.

The UI supports session-based chat history.

---

## Hybrid Inference Strategy

The system uses a hybrid approach for inference.

1. Rule-based routing for business-critical keywords
2. Machine learning classification for general queries
3. Confidence thresholding for uncertain predictions
4. Fallback to human support when confidence is low

This improves reliability in production environments.

---

## Containerized Deployment

The application is containerized using Docker.

Docker Compose runs both services:

* FastAPI backend
* Streamlit frontend

To build and run the system:

```bash
docker-compose up --build
```

Access:

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

* Environment reproducibility
* Compatibility with serialized models
* Stable deployments

Torch CPU wheels are installed separately in Docker to ensure Linux compatibility.

---

## Monitoring and Logging

### Inference Logging

Each API request is logged with:

* Timestamp
* User query
* Predicted intent
* Confidence score

Logs are stored in:

```
monitoring/data/inference_log.csv
```

---

### Drift Detection

Evidently is used to compare training data with production logs.

Drift reports are generated using:

```bash
python monitoring/drift_report.py
```

Output:

```
monitoring/reports/drift.html
```

---

## Retraining Pipeline

When drift is detected or performance degrades, retraining can be triggered.

The retraining script:

1. Reloads dataset
2. Regenerates embeddings
3. Trains a new model
4. Logs results to MLflow
5. Updates model artifacts

Command:

```bash
python training/retrain.py
```

---

## Version Control and Reproducibility

The project uses:

* Git for source code
* DVC for datasets and models
* MLflow for experiments

This enables full reproducibility of training and deployment.

Any previous version of the system can be restored.

---

## Future Improvements

* Automated retraining using schedulers
* Online learning support
* Integration with cloud storage (S3, GCS)
* Model calibration
* LLM-based fallback responses
* User feedback loop

---

## How to Run Locally (Without Docker)

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

5. Start UI (new terminal)

```bash
streamlit run ui/app.py
```

---

## Author

Developed as a hands-on MLOps portfolio project focused on production-level machine learning systems.

---
