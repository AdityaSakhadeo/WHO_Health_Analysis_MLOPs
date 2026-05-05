# WHO Health Analysis MLOps

This project is an end-to-end Machine Learning Operations (MLOps) pipeline designed to predict Life Expectancy (or other WHO health indicators) based on various socio-economic, health, and demographic features.

It is structured to demonstrate industry best practices across the entire machine learning lifecycle, from data versioning to model deployment and monitoring.

## 🚀 Features & Stack

1. **Model Training & Serving**: 
   - `scikit-learn` for the Random Forest model and preprocessing pipelines.
   - `FastAPI` for high-performance model serving (`src/predictor.py`).
2. **Experiment Tracking & Model Registry**: 
   - `MLflow` tracks metrics, hyperparameters, data fingerprints, and model artifacts.
3. **Data Versioning**: 
   - `DVC` (Data Version Control) tracks the dataset (`Data/health_indicators.csv`).
4. **Orchestration**: 
   - `Apache Airflow` schedules and manages the data validation, training, and deployment pipeline.
5. **Observability**: 
   - `Prometheus` scrapes system and FastAPI metrics.
   - `Grafana` provides rich dashboards to monitor API health and traffic.
6. **CI/CD**: 
   - `GitHub Actions` handles Continuous Integration (testing) and Continuous Deployment (building Docker images).
7. **Containerization**: 
   - `Docker` and `Docker Compose` ensure reproducible environments across Airflow, Model Serving, and Observability.

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Git

### 1. Local Setup
```bash
# Clone the repository
git clone <repository_url>
cd WHO_Health_Analysis_MLOPs

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Manual Training & Inference
You can manually train the model and launch the inference API.
```bash
# Train the model (this creates artifacts and logs to local MLflow)
python src/predictor.py train --data-path Data/health_indicators.csv --out-dir artifacts

# Serve the model using FastAPI
python src/predictor.py serve --model-path artifacts/model.joblib
```
Your API will be running at `http://127.0.0.1:8000`. You can visit `http://127.0.0.1:8000/docs` to see the Swagger UI and test predictions.

---

## ⚙️ Running the Full Stack (Docker Compose)

### Orchestration (Airflow)
To spin up Airflow for pipeline scheduling:
```bash
docker-compose -f docker-compose.airflow.yml up -d
```
- Access the Airflow UI at `http://localhost:8080` (Credentials: `admin` / `admin`).
- Unpause the `who_health_pipeline` DAG to trigger a training run automatically.

### Observability (Prometheus & Grafana)
To spin up the monitoring stack:
```bash
docker-compose -f docker-compose.observability.yml up -d
```
- Access **Grafana** at `http://localhost:3000` (Credentials: `admin` / `admin`). The Prometheus data source is auto-provisioned.
- Access **Prometheus** at `http://localhost:9090`.

*(Note: Make sure your FastAPI predictor server is running so Prometheus has an endpoint to scrape metrics from).*

---

## 🧪 Testing
Run the test suite using `pytest`:
```bash
pytest tests/
```

## 📦 Data Versioning (DVC)
If you add new data, track it using DVC:
```bash
dvc add Data/health_indicators.csv
git add Data/health_indicators.csv.dvc
git commit -m "Update health indicators data"
dvc push  # If a remote is configured
```
