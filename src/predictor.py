from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from matplotlib import pyplot as plt
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# Experiment config knobs
# =========================
# Change these in code to create a new tracked experiment run.
DATA_PATH = Path("Data/health_indicators.csv")
TARGET_COL = "life_expectancy"

# If empty, features are inferred as: all columns except TARGET_COL (and obvious identifiers).
# If you set this list explicitly, it will be logged to MLflow as a param to track changes.
FEATURES: List[str] = [
    # demographics / econ / infra / risk factors (example starter set)
    "population",
    "population_growth",
    "urban_pct",
    "death_rate",
    "birth_rate",
    "health_expenditure_pct_gdp",
    "health_expenditure_per_capita",
    "physicians_per_1000",
    "hospital_beds_per_1000",
    "nurses_per_1000",
    "under5_mortality",
    "neonatal_mortality",
    "maternal_mortality",
    "tb_incidence",
    "hiv_incidence",
    "communicable_death_pct",
    "noncommunicable_death_pct",
    "smoking_male",
    "smoking_female",
    "obesity_pct",
    "alcohol_per_capita",
    "diabetes_pct",
    "safe_water_pct",
    "sanitation_pct",
    "handwashing_pct",
    "immunization_measles",
    "immunization_dpt",
    "immunization_bcg",
    "immunization_pol3",
    "immunization_hib3",
    "stunting_pct",
    "wasting_pct",
    "overweight_children_pct",
    "undernourishment_pct",
    "fertility_rate",
    "adolescent_fertility",
    "prenatal_care_pct",
    "gdp_per_capita",
    "gdp_per_capita_ppp",
    "poverty_rate",
    # a couple categoricals (kept here on purpose to demonstrate feature tracking)
    "region",
    "income_level",
]

# Identifiers we never want as model inputs unless explicitly asked.
DEFAULT_DROP_COLS = {"country_code", "country_name"}


def safe_set_experiment(name: str) -> str:
    """
    MLflow raises if an experiment exists but is deleted. In that case we create a new name.
    Returns the actual experiment name that was activated/created.
    """
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None and getattr(exp, "lifecycle_stage", None) == "deleted":
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"{name}-{ts}"
    mlflow.set_experiment(name)
    return name


@dataclass(frozen=True)
class DataFingerprint:
    path: str
    sha256: str
    bytes: int
    modified_utc: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_data(path: Path) -> DataFingerprint:
    st = path.stat()
    return DataFingerprint(
        path=str(path.as_posix()),
        sha256=sha256_file(path),
        bytes=st.st_size,
        modified_utc=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    )


def infer_features(df: pd.DataFrame, target: str) -> List[str]:
    return [c for c in df.columns if c != target and c not in DEFAULT_DROP_COLS]


def build_pipeline(X: pd.DataFrame, *, random_state: int) -> Pipeline:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocess", pre), ("model", model)])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_frame(df: pd.DataFrame, *, target_col: str, feature_cols: List[str]) -> Dict[str, Any]:
    issues: List[str] = []
    if target_col not in df.columns:
        issues.append(f"missing_target:{target_col}")
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        issues.append(f"missing_features:{missing_features}")

    summary: Dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "issues": issues,
    }
    return summary


def missingness_summary(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    s = df[cols].isna().mean().sort_values(ascending=False)
    return pd.DataFrame({"feature": s.index, "missing_frac": s.values})


def _get_expanded_feature_names(preprocess: ColumnTransformer) -> List[str]:
    # sklearn >=1.0 should support get_feature_names_out, but we guard anyway.
    try:
        names = list(preprocess.get_feature_names_out())
        return [str(n) for n in names]
    except Exception:
        return []


def log_feature_importance_artifacts(pipe: Pipeline, *, out_dir: Path, top_k: int = 25) -> None:
    preprocess: ColumnTransformer = pipe.named_steps["preprocess"]
    model: RandomForestRegressor = pipe.named_steps["model"]

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return

    feat_names = _get_expanded_feature_names(preprocess)
    if not feat_names or len(feat_names) != len(importances):
        feat_names = [f"f{i}" for i in range(len(importances))]

    df_imp = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "feature_importance.csv"
    df_imp.to_csv(csv_path, index=False)
    mlflow.log_artifact(str(csv_path))

    top = df_imp.head(top_k)
    fig_h = max(4.0, 0.25 * float(len(top)))
    plt.figure(figsize=(10, fig_h))
    plt.barh(list(reversed(top["feature"].tolist())), list(reversed(top["importance"].tolist())))
    plt.xlabel("Importance")
    plt.title(f"Top {min(top_k, len(df_imp))} Feature Importances")
    plt.tight_layout()
    png_path = out_dir / "feature_importance_top.png"
    plt.savefig(png_path, dpi=160)
    plt.close()
    mlflow.log_artifact(str(png_path))


def train(
    *,
    data_path: Path,
    target_col: str,
    features: Optional[List[str]],
    experiment_name: str,
    run_name: Optional[str],
    random_state: int,
    test_size: float,
    output_dir: Path,
    feature_set_name: str = "default",
    dataset_version: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    enable_sweep: bool = False,
    sweep_n_iter: int = 15,
    sweep_cv: int = 3,
) -> tuple[Path, Dict[str, float]]:
    df = load_frame(data_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    feats = features if features and len(features) > 0 else infer_features(df, target_col)
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns missing from dataset: {missing}")

    work = df[feats + [target_col]].copy()
    work = work.dropna(subset=[target_col])

    X = work[feats]
    y = work[target_col].astype(float).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipe = build_pipeline(X_train, random_state=random_state)

    actual_experiment = safe_set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        fp = fingerprint_data(data_path)

        if actual_experiment != experiment_name:
            mlflow.log_param("requested_experiment", experiment_name)
            mlflow.log_param("actual_experiment", actual_experiment)

        mlflow.log_dict(asdict(fp), "data_fingerprint.json")
        mlflow.log_param("data_path", str(data_path.as_posix()))
        if dataset_version is not None:
            mlflow.log_param("dataset_version", str(dataset_version))
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("features", json.dumps(feats, ensure_ascii=False))
        mlflow.log_param("n_features", len(feats))
        mlflow.log_param("feature_set_name", feature_set_name)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("enable_sweep", bool(enable_sweep))
        mlflow.log_param("sweep_n_iter", int(sweep_n_iter))
        mlflow.log_param("sweep_cv", int(sweep_cv))

        # data stats
        mlflow.log_metric("train_rows", float(X_train.shape[0]))
        mlflow.log_metric("test_rows", float(X_test.shape[0]))

        # basic validation report + missingness summary
        val = validate_frame(df, target_col=target_col, feature_cols=feats)
        mlflow.log_dict(val, "validation_report.json")

        miss_df = missingness_summary(work, feats + [target_col])
        miss_path = output_dir / "missingness_summary.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        miss_df.to_csv(miss_path, index=False)
        mlflow.log_artifact(str(miss_path))

        if model_params:
            pipe.named_steps["model"].set_params(**model_params)

        model: RandomForestRegressor = pipe.named_steps["model"]
        mlflow.log_params(
            {
                "model_type": "RandomForestRegressor",
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "min_samples_split": model.min_samples_split,
                "min_samples_leaf": model.min_samples_leaf,
                "max_features": model.max_features,
            }
        )

        if enable_sweep:
            # sweep only the estimator hyperparams; preprocessing stays fixed.
            search_space = {
                "model__n_estimators": [200, 400, 800],
                "model__max_depth": [None, 6, 10, 16, 24],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            }

            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=search_space,
                n_iter=int(sweep_n_iter),
                cv=int(sweep_cv),
                scoring="neg_root_mean_squared_error",
                random_state=random_state,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            pipe = search.best_estimator_

            mlflow.log_param("best_params", json.dumps(search.best_params_))
            mlflow.log_metric("cv_best_neg_rmse", float(search.best_score_))

            cv_path = output_dir / "cv_results.csv"
            pd.DataFrame(search.cv_results_).to_csv(cv_path, index=False)
            mlflow.log_artifact(str(cv_path))
        else:
            pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        metrics = evaluate(y_test, pred)
        mlflow.log_metrics(metrics)

        # feature importance (after fit)
        log_feature_importance_artifacts(pipe, out_dir=output_dir)

        model_path = output_dir / "model.joblib"
        joblib.dump(
            {
                "pipeline": pipe,
                "target_col": target_col,
                "features": feats,
                "trained_at_utc": datetime.now(tz=timezone.utc).isoformat(),
                "data_fingerprint": asdict(fp),
                "feature_set_name": feature_set_name,
                "dataset_version": dataset_version,
            },
            model_path,
        )
        mlflow.log_artifact(str(model_path))

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="sklearn_model",
            input_example=X_test.head(3),
        )

        return model_path, metrics


def load_model_bundle(model_path: Path) -> Dict[str, Any]:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("Invalid model bundle. Expected a dict with key 'pipeline'.")
    return bundle


def predict_one(model_bundle: Dict[str, Any], payload: Dict[str, Any]) -> float:
    pipe: Pipeline = model_bundle["pipeline"]
    feats: List[str] = model_bundle["features"]
    X = pd.DataFrame([{k: payload.get(k, None) for k in feats}])
    pred = pipe.predict(X)
    return float(pred[0])


def make_app(model_path: Path) -> FastAPI:
    bundle = load_model_bundle(model_path)

    class PredictRequest(BaseModel):
        data: Dict[str, Any]

    class PredictResponse(BaseModel):
        prediction: float
        target: str
        features: List[str]

    app = FastAPI(title="WHO Health Predictor", version="1.0")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "model_path": str(model_path.as_posix()),
            "target": bundle.get("target_col"),
            "n_features": len(bundle.get("features", [])),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        y = predict_one(bundle, req.data)
        return PredictResponse(
            prediction=y,
            target=str(bundle.get("target_col")),
            features=list(bundle.get("features", [])),
        )

    return app


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WHO health predictor with MLflow tracking.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train model and log an MLflow run.")
    p_train.add_argument("--data-path", type=str, default=str(DATA_PATH.as_posix()))
    p_train.add_argument("--target", type=str, default=TARGET_COL)
    p_train.add_argument(
        "--use-code-features",
        action="store_true",
        help="Use FEATURES list from code (logs it for tracking).",
    )
    p_train.add_argument("--experiment", type=str, default="who-health")
    p_train.add_argument("--run-name", type=str, default=None)
    p_train.add_argument("--random-state", type=int, default=42)
    p_train.add_argument("--test-size", type=float, default=0.2)
    p_train.add_argument("--out-dir", type=str, default="artifacts")
    p_train.add_argument("--feature-set-name", type=str, default="default")
    p_train.add_argument("--dataset-version", type=str, default=None)
    p_train.add_argument(
        "--model-params-json",
        type=str,
        default=None,
        help='Optional JSON dict for RF params (e.g. {"n_estimators": 600, "max_depth": 12}).',
    )
    p_train.add_argument("--sweep", action="store_true", help="Run a hyperparameter sweep (RandomizedSearchCV).")
    p_train.add_argument("--sweep-n-iter", type=int, default=15)
    p_train.add_argument("--sweep-cv", type=int, default=3)

    p_pred = sub.add_parser("predict", help="Predict using a saved model bundle.")
    p_pred.add_argument("--model-path", type=str, required=True)
    p_pred.add_argument(
        "--json",
        type=str,
        required=True,
        help='JSON object with feature values. Example: {"gdp_per_capita": 1234, "region": "South Asia"}',
    )

    p_serve = sub.add_parser("serve", help="Serve the model via FastAPI.")
    p_serve.add_argument("--model-path", type=str, required=True)
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.cmd == "train":
        data_path = Path(args.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {data_path}")
        feats = FEATURES if args.use_code_features else None
        model_params = json.loads(args.model_params_json) if args.model_params_json else None
        model_path, metrics = train(
            data_path=data_path,
            target_col=args.target,
            features=feats,
            experiment_name=args.experiment,
            run_name=args.run_name,
            random_state=args.random_state,
            test_size=args.test_size,
            output_dir=Path(args.out_dir),
            feature_set_name=args.feature_set_name,
            dataset_version=args.dataset_version,
            model_params=model_params,
            enable_sweep=bool(args.sweep),
            sweep_n_iter=int(args.sweep_n_iter),
            sweep_cv=int(args.sweep_cv),
        )
        print(json.dumps({"model_path": str(model_path.as_posix()), "metrics": metrics}, indent=2))
        return 0

    if args.cmd == "predict":
        bundle = load_model_bundle(Path(args.model_path))
        payload = json.loads(args.json)
        y = predict_one(bundle, payload)
        print(json.dumps({"prediction": y, "target": bundle.get("target_col")}, indent=2))
        return 0

    if args.cmd == "serve":
        import uvicorn

        app = make_app(Path(args.model_path))
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())

