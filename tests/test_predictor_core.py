import pandas as pd

from src.predictor import DEFAULT_DROP_COLS, infer_features, validate_frame


def test_infer_features_drops_target_and_default_ids() -> None:
    df = pd.DataFrame(
        [
            {
                "country_code": "IN",
                "country_name": "India",
                "life_expectancy": 70.0,
                "gdp_per_capita": 1234.0,
                "region": "South Asia",
            }
        ]
    )
    feats = infer_features(df, target="life_expectancy")
    assert "life_expectancy" not in feats
    for c in DEFAULT_DROP_COLS:
        assert c not in feats
    assert set(feats) == {"gdp_per_capita", "region"}


def test_validate_frame_reports_missing_target_and_features() -> None:
    df = pd.DataFrame([{"a": 1, "b": 2}])
    rep = validate_frame(df, target_col="y", feature_cols=["a", "c"])
    assert rep["n_rows"] == 1
    assert rep["n_cols"] == 2
    assert "missing_target:y" in rep["issues"]
    assert "missing_features:['c']" in rep["issues"]

