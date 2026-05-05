from __future__ import annotations

import json
from typing import Any

from prefect import flow, task
from prefect.logging import get_run_logger

from src.orchestration_steps import (
    deploy_step,
    ingest_step,
    register_step,
    train_evaluate_step,
    validate_step,
)


@task(name="ingest", retries=0)
def ingest_task(data_path: str) -> dict[str, Any]:
    return ingest_step(data_path=data_path)


@task(name="validate", retries=0)
def validate_task(data_path: str, target_col: str, use_code_features: bool) -> dict[str, Any]:
    return validate_step(data_path=data_path, target_col=target_col, use_code_features=use_code_features)


@task(name="train_evaluate", retries=0)
def train_evaluate_task(
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
    return train_evaluate_step(
        data_path=data_path,
        target_col=target_col,
        use_code_features=use_code_features,
        experiment_name=experiment_name,
        run_name=run_name,
        out_dir=out_dir,
        feature_set_name=feature_set_name,
        dataset_version=dataset_version,
        model_params_json=model_params_json,
        sweep=sweep,
        sweep_n_iter=sweep_n_iter,
        sweep_cv=sweep_cv,
    )


@task(name="register", retries=0)
def register_task(registered_model_name: str) -> dict[str, Any]:
    return register_step(model_name=registered_model_name)


@task(name="deploy", retries=0)
def deploy_task() -> dict[str, Any]:
    return deploy_step()


@flow(name="who_health_mlops_pipeline")
def who_health_mlops_pipeline(
    *,
    data_path: str = "Data/health_indicators.csv",
    target_col: str = "life_expectancy",
    use_code_features: bool = True,
    experiment_name: str = "who-health",
    run_name: str = "prefect_run",
    out_dir: str = "artifacts",
    feature_set_name: str = "fs_prefect",
    dataset_version: str | None = "dvc",
    model_params_json: str | None = None,
    sweep: bool = False,
    sweep_n_iter: int = 15,
    sweep_cv: int = 3,
    registered_model_name: str = "who-health-rf",
) -> dict[str, Any]:
    """
    Pipeline order:
    ingest -> validate -> train_evaluate -> register -> deploy
    """
    logger = get_run_logger()
    logger.info("Starting pipeline run_name=%s experiment=%s", run_name, experiment_name)

    if model_params_json is None:
        model_params_json = json.dumps({"n_estimators": 400})

    ingest = ingest_task(data_path)
    validation = validate_task(data_path, target_col, use_code_features)

    train_eval = train_evaluate_task(
        data_path=data_path,
        target_col=target_col,
        use_code_features=use_code_features,
        experiment_name=experiment_name,
        run_name=run_name,
        out_dir=out_dir,
        feature_set_name=feature_set_name,
        dataset_version=dataset_version,
        model_params_json=model_params_json,
        sweep=sweep,
        sweep_n_iter=sweep_n_iter,
        sweep_cv=sweep_cv,
    )

    registered = register_task(registered_model_name)
    deployed = deploy_task()

    summary = {
        "ingest": ingest,
        "validation": validation,
        "train_evaluate": train_eval,
        "register": registered,
        "deploy": deployed,
    }
    logger.info("Pipeline completed.")
    return summary


if __name__ == "__main__":
    who_health_mlops_pipeline()

