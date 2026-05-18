from __future__ import annotations

from typing import Any

import bentoml
from pydantic import BaseModel

from src.predictor import predict_one


class PredictResponse(BaseModel):
    prediction: float
    target: str
    features: list[str]


MODEL_NAME = "who_health_model"


@bentoml.service(name="who_health_predictor")
class WhoHealthService:
    def __init__(self) -> None:
        # BentoML v1.4+: resolve latest model via generic model store API.
        self.model_ref = bentoml.models.get(f"{MODEL_NAME}:latest")
        self.model_bundle = bentoml.picklable_model.load_model(self.model_ref)

    @bentoml.api
    def predict(self, data: dict[str, Any]) -> PredictResponse:
        y = predict_one(self.model_bundle, data)
        return PredictResponse(
            prediction=y,
            target=str(self.model_bundle.get("target_col")),
            features=list(self.model_bundle.get("features", [])),
        )

    @bentoml.api
    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "target": self.model_bundle.get("target_col"),
            "n_features": len(self.model_bundle.get("features", [])),
            "model_tag": str(self.model_ref.tag),
        }

