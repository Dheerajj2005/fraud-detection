# 💳 Credit Card Fraud Detection - End-to-End MLOps Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready Machine Learning system built to detect fraudulent credit card transactions.

This project is explicitly designed to demonstrate **modern MLOps principles, scalable architecture, and engineering best practices**. It goes significantly beyond simply training a model in a Jupyter Notebook by implementing a robust end-to-end lifecycle encompassing data ingestion, experiment tracking, model serving, and continuous monitoring.

---

## 🚀 Key MLOps Features

This project utilizes a modern MLOps stack to ensure reproducibility, reliability, and observability of the machine learning system:

* **Experiment Tracking & Model Registry (`MLflow`)**: Manages model parameters, metrics (Precision, Recall, F1), and artifacts. Models are versioned and stored systematically.
* **Orchestration & Workflow (`Prefect`)**: Schedules, monitors, and orchestrates data pipelines and model training workflows to ensure reliable execution.
* **High-Performance Model Serving (`FastAPI` & `Uvicorn`)**: Deploys the best-performing model behind a fast, highly concurrent, and documented REST API.
* **Containerization (`Docker` & `Docker Compose`)**: Encapsulates the API, Monitoring tools, and dependencies for seamless local testing and multi-environment deployment.
* **Monitoring & Observability (`Prometheus`, `Grafana`, & `Evidently AI`)**:
  * Exposes real-time API performance metrics via Prometheus.
  * Visualizes system health and load in Grafana dashboards.
  * Tracks data drift and ML model performance degradations in production using Evidently AI.
* **Robust Testing (`Pytest` & `Locust`)**: Unit and integration tests ensure code reliability, while Locust is used to load test the FastAPI endpoints under concurrent traffic.
* **Code Quality & CI (`Pre-commit`, `Black`, `Flake8`, `Mypy`)**: Enforces strict typing, aggressive linting, and automated code formatting before commits are allowed.
* **Interactive UI (`Streamlit`)**: A sleek, user-friendly frontend enabling business users to run predictions and interact with historical data.

---

## 📁 Project Architecture

```plaintext
fraud-detection/
├── api/                  # FastAPI real-time model serving logic
├── config/               # Configuration files (YAML, JSON)
├── data/                 # Raw and Processed datasets
├── mlruns/               # MLflow experiment tracking logs & artifacts
├── models/               # Serialized finalized models used for serving
├── monitoring/           # Prometheus, Grafana, & Evidently configs
├── pipelines/            # Prefect data and ML pipelines
├── reports/              # Model performance and automated data drift reports
├── src/                  # Core library functions: training, data logic, inference
├── streamlit_app/        # Interactive business UI (Prediction & History pages)
├── tests/                # Pytest unit and integration tests
├── Dockerfile            # Packaging the API as an isolated Docker image
├── docker-compose.yml    # Orchestrates API, Prometheus, and Grafana containers
└── requirements.txt      # Comprehensive, locked Python dependencies
```

---

## 🛠️ Quickstart

### 1. Environment Setup
Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/Dheerajj2005/fraud-detection.git
cd fraud-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Full Stack (API + Monitoring)
We use Docker Compose to start up the FastAPI serving layer, Prometheus for metrics scraping, and Grafana for visualization:

```bash
docker-compose up -d --build
```
* **FastAPI Docs (Swagger UI)**: `http://localhost:8000/docs`
* **Prometheus**: `http://localhost:9090`
* **Grafana**: `http://localhost:3000` *(Default Login: admin / admin)*

### 3. Run the Streamlit User Interface
To give business users a quick way to interact with the deployed model and analyze history:

```bash
streamlit run streamlit_app/pages/1_Prediction.py
```
* **Streamlit App**: `http://localhost:8501`

### 4. Run Tests & Validation
Execute unit tests, verify code coverage, and check code formatting:

```bash
# Run tests with coverage
pytest --cov=src --cov=api tests/

# Trigger load testing against API
locust -f tests/test_api.py --host=http://localhost:8000
```

---

## 📡 API Reference

The serving layer handles dynamic JSON payloads. Example API request to the `/predict` endpoint:

```json
{
  "Time": 40000,
  "V1": -1.5, "V2": 2.1, "V3": -2.3, "V4": 1.2, "V5": -1.8,
  "V6": -0.5, "V7": 1.9, "V8": 0.4, "V9": -2.0, "V10": -3.1,
  "V11": 2.2, "V12": -1.7, "V13": 0.6, "V14": -2.8, "V15": -0.9,
  "V16": -1.4, "V17": -3.5, "V18": -1.1, "V19": 0.2, "V20": 1.1,
  "V21": 0.3, "V22": 0.8, "V23": -0.4, "V24": 0.7, "V25": -0.6,
  "V26": 0.1, "V27": 0.9, "V28": -0.2,
  "Amount": 5000
}
```

#### Example Request in PowerShell:
```powershell
Invoke-RestMethod `
  -Uri http://localhost:8000/predict `
  -Method POST `
  -ContentType application/json `
  -Body '{
    "Time": 40000, "V1": -1.5, "V2": 2.1, "V3": -2.3, "V4": 1.2, "V5": -1.8,
    "V6": -0.5, "V7": 1.9, "V8": 0.4, "V9": -2.0, "V10": -3.1,
    "V11": 2.2, "V12": -1.7, "V13": 0.6, "V14": -2.8, "V15": -0.9,
    "V16": -1.4, "V17": -3.5, "V18": -1.1, "V19": 0.2, "V20": 1.1,
    "V21": 0.3, "V22": 0.8, "V23": -0.4, "V24": 0.7, "V25": -0.6,
    "V26": 0.1, "V27": 0.9, "V28": -0.2,
    "Amount": 5000
  }'
```

---

## 🔮 Future Roadmap

* **CI/CD Pipelines**: Adding automated GitHub Actions workflows for continuous integration (testing, linting) and continuous deployment (container pushing).
* **Cloud Deployment**: Migration and managed hosting on AWS EKS or GCP Cloud Run for infinite horizontal scalability.
* **Feature Store Integration**: Implementing tools like Feast or Hopsworks to manage and retrieve real-time features efficiently.
