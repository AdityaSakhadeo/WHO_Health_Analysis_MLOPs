from __future__ import annotations

import json
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.orchestration_steps import (
    deploy_step,
    ingest_step,
    register_step,
    train_evaluate_step,
    validate_step,
)

"""
Airflow DAG: ingest -> validate -> train -> evaluate -> register -> deploy

Parameterization:
- feature_set_name
- dataset_version
- model hyperparameters (JSON)

Notes:
- This DAG assumes the repo is mounted/available to the Airflow worker.
- Airflow itself is not added to requirements.txt because it is commonly managed via a dedicated
  Airflow environment/container.
"""

default_args = {"owner": "mlops", "retries": 0}


with DAG(
    dag_id="who_health_mlops_pipeline",
    default_args=default_args,
    description="WHO health indicators pipeline (MLflow + DVC ready)",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    params={
        "data_path": "Data/health_indicators.csv",
        "target_col": "life_expectancy",
        "use_code_features": True,
        "experiment_name": "who-health",
        "run_name": "airflow_run",
        "out_dir": "artifacts",
        "feature_set_name": "fs_airflow",
        "dataset_version": "dvc",
        "model_params_json": json.dumps({"n_estimators": 400}),
        "sweep": False,
        "sweep_n_iter": 15,
        "sweep_cv": 3,
        "registered_model_name": "who-health-rf",
    },
    render_template_as_native_obj=True,
) as dag:
    ingest = PythonOperator(
        task_id="ingest",
        python_callable=ingest_step,
        op_kwargs={"data_path": "{{ params.data_path }}"},
    )

    validate = PythonOperator(
        task_id="validate",
        python_callable=validate_step,
        op_kwargs={
            "data_path": "{{ params.data_path }}",
            "target_col": "{{ params.target_col }}",
            "use_code_features": "{{ params.use_code_features }}",
        },
    )

    train_eval = PythonOperator(
        task_id="train_evaluate",
        python_callable=train_evaluate_step,
        op_kwargs={
            "data_path": "{{ params.data_path }}",
            "target_col": "{{ params.target_col }}",
            "use_code_features": "{{ params.use_code_features }}",
            "experiment_name": "{{ params.experiment_name }}",
            "run_name": "{{ params.run_name }}",
            "out_dir": "{{ params.out_dir }}",
            "feature_set_name": "{{ params.feature_set_name }}",
            "dataset_version": "{{ params.dataset_version }}",
            "model_params_json": "{{ params.model_params_json }}",
            "sweep": "{{ params.sweep }}",
            "sweep_n_iter": "{{ params.sweep_n_iter }}",
            "sweep_cv": "{{ params.sweep_cv }}",
        },
    )

    register = PythonOperator(
        task_id="register",
        python_callable=register_step,
        op_kwargs={"model_name": "{{ params.registered_model_name }}"},
    )

    deploy = PythonOperator(task_id="deploy", python_callable=deploy_step)

    ingest >> validate >> train_eval >> register >> deploy
