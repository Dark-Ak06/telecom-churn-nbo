import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
import pytest

from src.build_features import build_features
from src.generate_data import generate_data
from src.train_churn import FEATURES, NUMERICAL, evaluate_model, temporal_split, train_churn


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple:
    tmp_path = tmp_path_factory.mktemp("churn")
    raw = generate_data(tmp_path / "raw.parquet", n_users=12_000, as_of=date(2026, 11, 1))
    mart = build_features(raw, tmp_path / "mart.parquet", tmp_path / "eda.json")
    train, validation, test = temporal_split(pd.read_parquet(mart))
    validation, test = validation.copy(), test.copy()
    validation.loc[validation.index[0], ["region", "arpu_kzt"]] = ["validation_only", 50_000]
    test.loc[test.index[0], ["region", "arpu_kzt"]] = ["test_only", 90_000]
    metrics = train_churn(train, validation, test, artifacts_dir=tmp_path)
    return train, validation, test, metrics, tmp_path


def test_temporal_split_and_training_reject_overlapping_labels(trained: tuple) -> None:
    train, validation, test, _, path = trained
    assert [part["month"].nunique() for part in (train, validation, test)] == [4, 1, 1]
    assert train["month"].max() < validation["month"].min() < test["month"].min()
    assert not {"historical_offer", "offer_accepted", "churn", "user_id", "month"} & set(FEATURES)
    with pytest.raises(ValueError, match="labels must be mature"):
        train_churn(pd.concat([train, validation]), validation, test, artifacts_dir=path)


def test_temporal_split_rejects_missing_month() -> None:
    frame = pd.DataFrame({"month": [date(2026, month, 1) for month in (1, 2, 3, 5, 6, 7)]})
    with pytest.raises(ValueError, match="consecutive observation months"):
        temporal_split(frame)


def test_preprocessing_is_fitted_on_training_rows_only(trained: tuple) -> None:
    train, _, _, _, path = trained
    model = joblib.load(path / "churn_model.joblib")
    preprocessing = model.named_steps["preprocess"].named_transformers_
    np.testing.assert_allclose(preprocessing["numerical"].mean_, train[NUMERICAL].mean())
    assert set(preprocessing["categorical"].categories_[0]) == set(train["region"])


def test_holdout_predictions_and_metrics_are_finite(trained: tuple) -> None:
    _, _, test, metrics, path = trained
    winner = metrics["models"][metrics["best_model"]]
    assert 0.80 <= winner["roc_auc"] <= 0.87
    assert winner["lift_at_10_pct"] > 2.5
    assert len(metrics["permutation_importance"]) == 12
    for scores in metrics["models"].values():
        matrix = scores["confusion_matrix"]
        assert sum(matrix[0]) + sum(matrix[1]) == len(test)
        assert matrix[0][1] + matrix[1][1] == len(test) // 10
        assert scores["calibration_curve"]["mean_predicted_probability"]
    model = joblib.load(path / "churn_model.joblib")
    probabilities = model.predict_proba(test[FEATURES])
    assert probabilities.shape == (len(test), 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="finite churn probability"):
        evaluate_model(test, np.full(len(test), np.inf))
    # Strict serialization validates nested scalars and all points on all curves.
    json.dumps(metrics, allow_nan=False)
