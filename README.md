# WHO Health MLOps Project (Prefect Orchestration)

This project trains and serves a **life expectancy** predictor using **scikit-learn** with **MLflow** tracking.
Pipeline orchestration is done with **Prefect** (Python-native).

## Why Prefect in this repo

Prefect is used to orchestrate the end-to-end ML workflow defined in `src/orchestration_steps.py`:

- **`ingest_step`**: loads the dataset and records basic stats
- **`validate_step`**: validates required columns/features and (optionally) logs a report to MLflow
- **`train_evaluate_step`**: trains + evaluates the model and logs artifacts/metrics to MLflow
- **`register_step`**: placeholder for MLflow Model Registry integration
- **`deploy_step`**: placeholder for deployment (e.g., building & deploying the FastAPI service)

With Prefect you get:

- **Local-first execution** (run flows as plain Python)
- **Parameterization** (same pipeline, different inputs/hyperparams)
- **Observability** (task/flow run history + logs in the Prefect UI)
- **Retries / caching / scheduling** if you later want it (without rewriting the pipeline)

## Setup

From the repo root:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Prefect pipeline (no server required)

This runs the pipeline locally as a normal Python program:

```bash
python prefect_pipeline.py
```

## Run with Prefect UI (recommended)

1) Start Prefect server (in one terminal):

```bash
prefect server start
```

2) In another terminal, point Prefect to the local server and run:

```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
python prefect_pipeline.py
```

Then open the UI at `http://127.0.0.1:4200` and you’ll see the flow + task runs.

## Change pipeline parameters

Edit defaults in `prefect_pipeline.py` or call the flow from Python:

```python
from prefect_pipeline import who_health_mlops_pipeline

who_health_mlops_pipeline(
    data_path="Data/health_indicators.csv",
    experiment_name="who-health",
    run_name="my_run",
    model_params_json='{"n_estimators": 600, "max_depth": 12}',
    sweep=False,
)
```

## Run tests

```bash
pytest -q
```

