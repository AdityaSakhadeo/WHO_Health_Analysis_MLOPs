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
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# Experiment config knobs
# =========================
# Change these in code to create a new tracked experiment run.
DATA_PATH = Path("data/health_indicators.csv")
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

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        fp = fingerprint_data(data_path)

        mlflow.log_dict(asdict(fp), "data_fingerprint.json")
        mlflow.log_param("data_path", str(data_path.as_posix()))
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("features", json.dumps(feats, ensure_ascii=False))
        mlflow.log_param("n_features", len(feats))
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)

        model: RandomForestRegressor = pipe.named_steps["model"]
        mlflow.log_params(
            {
                "model_type": "RandomForestRegressor",
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "min_samples_split": model.min_samples_split,
                "min_samples_leaf": model.min_samples_leaf,
            }
        )

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = evaluate(y_test, pred)
        mlflow.log_metrics(metrics)

        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.joblib"
        joblib.dump(
            {
                "pipeline": pipe,
                "target_col": target_col,
                "features": feats,
                "trained_at_utc": datetime.now(tz=timezone.utc).isoformat(),
                "data_fingerprint": asdict(fp),
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
        model_path, metrics = train(
            data_path=data_path,
            target_col=args.target,
            features=feats,
            experiment_name=args.experiment,
            run_name=args.run_name,
            random_state=args.random_state,
            test_size=args.test_size,
            output_dir=Path(args.out_dir),
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

