# WHO Health MLOps Project

This project predicts life expectancy and includes:

- Git for version control
- DVC for dataset versioning
- MLflow for experiment tracking
- Prefect for pipeline orchestration
- BentoML for model serving
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

Use `prefect_pipeline.py` as your default execution path for training workflow.
Manual one-off commands are useful mainly for debugging.

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

## 5) Register model in BentoML model store

After training completes and `artifacts/model.joblib` is available:

```bash
python register_bento_model.py --model-path artifacts/model.joblib --name who_health_model
```

## 6) Serve inference with BentoML endpoint

Start Bento service:

```bash
bentoml serve bentoml_service:WhoHealthService --host 127.0.0.1 --port 3000
```

Predict request:

```bash
curl -X POST http://127.0.0.1:3000/predict -H "Content-Type: application/json" -d "{\"gdp_per_capita\":2500,\"region\":\"South Asia\"}"
```

Bento metrics endpoint:

```bash
curl http://127.0.0.1:3000/metrics
```

## 7) Run frontend app (input -> prediction)

In a separate terminal:

```bash
streamlit run frontend_app.py
```

Open: `http://127.0.0.1:8501`

## 8) Prometheus setup (without Docker)

1. Download Prometheus for Windows from [prometheus.io](https://prometheus.io/download/).
2. Extract it locally.
3. Replace its `prometheus.yml` with this repo config: `monitoring/prometheus.yml`.
4. Start Prometheus from the extracted folder:

```bash
.\prometheus.exe --config.file="D:\COEP Masters\Academics\SEM -2\MLOps Lab\MLOps_Project\monitoring\prometheus.yml"
```

Open: `http://127.0.0.1:9090`

## 9) Grafana setup (without Docker)

1. Download and install Grafana for Windows from [grafana.com](https://grafana.com/grafana/download).
2. Start Grafana service/app.
3. If BentoML is already using port `3000`, run Grafana on another port such as `3001`, then open that URL (default login: `admin` / `admin`).
4. Add Prometheus datasource:
   - URL: `http://127.0.0.1:9090`
5. Build dashboards from Bento metrics (check exact names at `http://127.0.0.1:3000/metrics`, usually prefixed with `bentoml_`).
   Typical useful panels:
   - request count / rate
   - request duration (p50/p95)
   - in-flight requests
   - non-2xx response counts

## 10) GitHub Actions

Workflows are in `.github/workflows/`:

- `ci.yml`: Ruff + Pytest on push/PR
- `cd-ghcr.yml`: builds and pushes image to GHCR on selected branches/tags

Push to your branch and check Actions tab for pipeline status.

## 11) Local quality checks

```bash
ruff check .
ruff format --check .
pytest -q
```

