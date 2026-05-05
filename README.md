# WHO Health MLOps Project

This project predicts life expectancy and includes:

- Git for version control
- DVC for dataset versioning
- MLflow for experiment tracking
- Prefect for pipeline orchestration
- GitHub Actions for CI/CD
- Prometheus + Grafana for runtime API monitoring

## Why Prefect in this repo

Prefect orchestrates the same ML workflow steps implemented in `src/orchestration_steps.py`:

- `ingest_step`
- `validate_step`
- `train_evaluate_step`
- `register_step`
- `deploy_step`

Prefect gives Python-native orchestration, parameterized flow runs, task-level logs, and an optional UI.

## 1) Initial setup (Windows / PowerShell)

From repository root:

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

## 2) DVC data pull

If your DVC remote is configured:

```bash
dvc pull
```

## 3) Run MLflow tracking UI

In a separate terminal:

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Open: `http://127.0.0.1:5000`

## 4) Run Prefect (pipeline orchestration)

### Option A: run flow directly

```bash
python prefect_pipeline.py
```

### Option B: run with Prefect UI

Terminal 1:

```bash
prefect server start
```

Terminal 2:

```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
python prefect_pipeline.py
```

Open: `http://127.0.0.1:4200`

## 5) Train and serve model API

Train (if needed):

```bash
python predictor.py train --data-path Data/health_indicators.csv --target life_expectancy --experiment who-health --run-name local_train --out-dir artifacts
```

Serve API:

```bash
python predictor.py serve --model-path artifacts/model.joblib --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prometheus metrics endpoint:

```bash
curl http://127.0.0.1:8000/metrics
```

## 6) Prometheus setup (without Docker)

1. Download Prometheus for Windows from [prometheus.io](https://prometheus.io/download/).
2. Extract it locally.
3. Replace its `prometheus.yml` with this repo config: `monitoring/prometheus.yml`.
4. Start Prometheus from the extracted folder:

```bash
.\prometheus.exe --config.file="D:\COEP Masters\Academics\SEM -2\MLOps Lab\MLOps_Project\monitoring\prometheus.yml"
```

Open: `http://127.0.0.1:9090`

## 7) Grafana setup (without Docker)

1. Download and install Grafana for Windows from [grafana.com](https://grafana.com/grafana/download).
2. Start Grafana service/app.
3. Open `http://127.0.0.1:3000` (default login: `admin` / `admin`).
4. Add Prometheus datasource:
   - URL: `http://127.0.0.1:9090`
5. Build dashboards using these metrics:
   - `who_health_http_requests_total`
   - `who_health_http_request_duration_seconds`

## 8) GitHub Actions

Workflows are in `.github/workflows/`:

- `ci.yml`: Ruff + Pytest on push/PR
- `cd-ghcr.yml`: builds and pushes image to GHCR on selected branches/tags

Push to your branch and check Actions tab for pipeline status.

## 9) Local quality checks

```bash
ruff check .
ruff format --check .
pytest -q
```

