from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow

from src.predictor import FEATURES, infer_features, load_frame, train, validate_frame


def ingest_step(data_path: str) -> dict[str, Any]:
    path = Path(data_path)
    df = load_frame(path)
    return {"data_path": str(path.as_posix()), "rows": int(df.shape[0]), "cols": int(df.shape[1])}


def validate_step(data_path: str, target_col: str, use_code_features: bool) -> dict[str, Any]:
    df = load_frame(Path(data_path))
    feats: list[str] = FEATURES if use_code_features else infer_features(df, target_col)
    rep = validate_frame(df, target_col=target_col, feature_cols=feats)
    # Optional: log validation into the currently-active MLflow run (if orchestration started one)
    try:
        if mlflow.active_run() is not None:
            mlflow.log_dict(rep, "validation_report_orchestrator.json")
            mlflow.log_param("orchestrator_use_code_features", bool(use_code_features))
            mlflow.log_param("orchestrator_features", json.dumps(feats, ensure_ascii=False))
    except Exception:
        pass
    return rep


def train_evaluate_step(
    *,
    data_path: str,
    target_col: str,
    use_code_features: bool,
    experiment_name: str,
    run_name: str,
    out_dir: str,
    feature_set_name: str,
    dataset_version: str | None,
    model_params_json: str | None,
    sweep: bool,
    sweep_n_iter: int,
    sweep_cv: int,
) -> dict[str, Any]:
    df = load_frame(Path(data_path))
    feats = FEATURES if use_code_features else infer_features(df, target_col)
    model_params = json.loads(model_params_json) if model_params_json else None
    model_path, metrics = train(
        data_path=Path(data_path),
        target_col=target_col,
        features=feats,
        experiment_name=experiment_name,
        run_name=run_name,
        random_state=42,
        test_size=0.2,
        output_dir=Path(out_dir),
        feature_set_name=feature_set_name,
        dataset_version=dataset_version,
        model_params=model_params,
        enable_sweep=bool(sweep),
        sweep_n_iter=int(sweep_n_iter),
        sweep_cv=int(sweep_cv),
    )
    return {"model_path": str(model_path.as_posix()), "metrics": metrics}


def register_step(model_name: str = "who-health-rf") -> dict[str, Any]:
    """
    Placeholder: in real projects you would register the MLflow model artifact to the registry.
    Keeping it minimal here because registry requires an MLflow backend that supports it.
    """
    return {
        "registered_model_name": model_name,
        "status": "skipped (configure MLflow registry backend to enable)",
    }


def deploy_step() -> dict[str, Any]:
    """
    Placeholder: deploy could mean building a docker image and deploying the FastAPI service.
    """
    return {"status": "skipped (deployment step placeholder)"}
