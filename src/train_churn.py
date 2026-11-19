"""Train and compare churn classifiers with a chronological holdout."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ARTIFACTS_DIR, N_MONTHS, RANDOM_SEED, TOP_FRACTION, write_json

CATEGORICAL = ["region", "tariff_plan", "rfm_segment"]
NUMERICAL = [
    "tenure_months",
    "arpu_kzt",
    "data_gb",
    "voice_min",
    "sms_count",
    "support_calls",
    "network_complaints",
    "days_since_last_topup",
    "app_sessions",
    "data_gb_mom_change",
    "app_sessions_mom_change",
    "data_gb_3m_avg",
    "arpu_3m_avg",
    "support_calls_cum",
    "has_previous_month",
]
FEATURES = CATEGORICAL + NUMERICAL


def temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = sorted(pd.unique(frame["month"]))
    if len(months) != N_MONTHS:
        raise ValueError(f"Expected {N_MONTHS} observation months, found {len(months)}")
    periods = pd.PeriodIndex(months, freq="M")
    if not periods.equals(pd.period_range(periods[0], periods=N_MONTHS, freq="M")):
        raise ValueError("Expected consecutive observation months")
    return (
        frame.loc[frame["month"].isin(months[:-2])],
        frame.loc[frame["month"].eq(months[-2])],
        frame.loc[frame["month"].eq(months[-1])],
    )


def require_mature_history(history: pd.DataFrame, scoring: pd.DataFrame) -> None:
    """Reject training labels that would still be open at the first scoring date."""
    if any(part.empty or part["month"].isna().any() for part in (history, scoring)):
        raise ValueError("Training and scoring require nonempty, dated snapshots")
    last_label_maturity = pd.Period(history["month"].max(), freq="M") + 2
    first_scoring_date = pd.Period(scoring["month"].min(), freq="M") + 1
    if last_label_maturity > first_scoring_date:
        raise ValueError("Training labels must be mature before scoring")


def classifier_pipeline(estimator: LogisticRegression | HistGradientBoostingClassifier) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
            ("numerical", StandardScaler(), NUMERICAL),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", estimator)])


def boosting_model(max_leaf_nodes: int = 31, max_iter: int = 130) -> Pipeline:
    return classifier_pipeline(
        HistGradientBoostingClassifier(
            max_leaf_nodes=max_leaf_nodes,
            max_iter=max_iter,
            min_samples_leaf=80,
            learning_rate=0.09,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=RANDOM_SEED,
        )
    )


def evaluate_model(test: pd.DataFrame, scores: np.ndarray) -> dict:
    actual = test["churn"].to_numpy()
    if (
        scores.shape != actual.shape
        or not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError("Expected one finite churn probability per test row")
    if np.unique(actual).size != 2:
        raise ValueError("Churn metrics require both classes in the test set")
    baseline = float(actual.mean())
    top_count = int(np.ceil(len(actual) * TOP_FRACTION))
    order = np.lexsort((test["user_id"].to_numpy(), -scores))
    top = order[:top_count]
    predicted = np.zeros(len(actual), dtype=int)
    predicted[top] = 1
    precision_at_10 = float(actual[top].mean())
    fpr, tpr, _ = roc_curve(actual, scores)
    precision, recall, _ = precision_recall_curve(actual, scores)
    fraction_positive, mean_predicted = calibration_curve(actual, scores, n_bins=10)
    return {
        "roc_auc": roc_auc_score(actual, scores),
        "pr_auc": average_precision_score(actual, scores),
        "precision_at_10_pct": precision_at_10,
        "recall_at_10_pct": float(actual[top].sum() / actual.sum()),
        "lift_at_10_pct": precision_at_10 / baseline,
        "top_count": top_count,
        "top_score_cutoff": float(scores[top[-1]]),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"recall": recall.tolist(), "precision": precision.tolist()},
        "calibration_curve": {
            "mean_predicted_probability": mean_predicted.tolist(),
            "fraction_positive": fraction_positive.tolist(),
        },
    }


def train_churn(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict:
    require_mature_history(train, validation)
    require_mature_history(validation, test)
    logistic = classifier_pipeline(LogisticRegression(max_iter=500, random_state=RANDOM_SEED)).fit(
        train[FEATURES], train["churn"]
    )

    candidates = {}
    validation_auc = {}
    for leaves in (15, 31):
        candidate = boosting_model(max_leaf_nodes=leaves).fit(train[FEATURES], train["churn"])
        validation_auc[leaves] = roc_auc_score(
            validation["churn"], candidate.predict_proba(validation[FEATURES])[:, 1]
        )
        candidates[leaves] = candidate

    best_leaves = max(validation_auc, key=validation_auc.__getitem__)
    models = {"LogisticRegression": logistic, "HistGradientBoosting": candidates[best_leaves]}
    validation_scores = {
        "LogisticRegression": roc_auc_score(
            validation["churn"], logistic.predict_proba(validation[FEATURES])[:, 1]
        ),
        "HistGradientBoosting": validation_auc[best_leaves],
    }
    winner = max(validation_scores, key=validation_scores.__getitem__)
    sample = test.sample(n=min(12_000, len(test)), random_state=RANDOM_SEED)
    importance = permutation_importance(
        models[winner],
        sample[FEATURES],
        sample["churn"],
        scoring="roc_auc",
        n_repeats=4,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    top_features = sorted(
        zip(FEATURES, importance.importances_mean, importance.importances_std, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )[:12]
    results = {
        "best_model": winner,
        "selection_metric": "validation_roc_auc",
        "boosting_max_leaf_nodes": best_leaves,
        "boosting_validation_roc_auc_by_max_leaf_nodes": validation_auc,
        "months": {
            name: sorted({str(month) for month in part["month"]})
            for name, part in (("train", train), ("validation", validation), ("test", test))
        },
        "test_rows": len(test),
        "test_target_month": str(pd.Period(test["month"].min(), freq="M") + 1),
        "test_churn_rate": float(test["churn"].mean()),
        "models": {
            name: {
                "validation_roc_auc": validation_scores[name],
                **evaluate_model(test, model.predict_proba(test[FEATURES])[:, 1]),
            }
            for name, model in models.items()
        },
        "permutation_importance_test_rows": len(sample),
        "permutation_importance": [
            {"feature": feature, "mean_auc_drop": float(mean), "std_auc_drop": float(std)}
            for feature, mean, std in top_features
        ],
    }
    write_json(artifacts_dir / "model_metrics.json", results)
    joblib.dump(models[winner], artifacts_dir / "churn_model.joblib")
    return results
