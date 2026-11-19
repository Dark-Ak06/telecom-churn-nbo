import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.build_features import build_features
from src.config import NO_OFFER, OFFERS
from src.generate_data import REGIONS, TARIFFS, generate_data
from src.train_churn import FEATURES


def test_data_schema_and_duckdb_windows(tmp_path: Path) -> None:
    raw_path = generate_data(tmp_path / "raw.parquet", n_users=3_000, as_of=date(2026, 11, 1))
    raw = pd.read_parquet(raw_path)
    assert list(raw.columns) == [
        "user_id",
        "month",
        "region",
        "tariff_plan",
        "tenure_months",
        "arpu_kzt",
        "data_gb",
        "voice_min",
        "sms_count",
        "support_calls",
        "network_complaints",
        "days_since_last_topup",
        "app_sessions",
        "historical_offer",
        "offer_accepted",
        "churn",
    ]
    assert pa.types.is_date32(pq.read_schema(raw_path).field("month").type)
    assert pd.api.types.is_integer_dtype(raw["user_id"])
    assert raw["user_id"].nunique() == 3_000
    assert len(raw) == 3_000 * 6
    assert not raw.duplicated(["user_id", "month"]).any()
    assert list(raw["month"].unique()) == [date(2026, month, 1) for month in range(4, 10)]
    label_maturity = pd.Period(raw["month"].max(), freq="M") + 2
    assert label_maturity.start_time.date() == date(2026, 11, 1)
    assert set(raw["region"]) == set(REGIONS)
    assert set(raw["tariff_plan"]) == set(TARIFFS)
    assert set(raw["historical_offer"]) == set(OFFERS)
    assert set(raw["churn"]) == set(raw["offer_accepted"]) == {0, 1}
    assert raw.loc[raw["historical_offer"].eq(NO_OFFER), "offer_accepted"].eq(0).all()
    assert raw["arpu_kzt"].between(2500, 12000).all()
    assert raw.groupby("month")["churn"].mean().between(0.05, 0.30, inclusive="neither").all()

    eda_path = tmp_path / "eda.json"
    mart_path = build_features(raw_path, tmp_path / "mart.parquet", eda_path)
    mart = pd.read_parquet(mart_path).sort_values(["user_id", "month"])
    grouped = mart.groupby("user_id", sort=False)
    assert len(mart) == len(raw)
    assert not mart.isna().any().any()
    assert set(mart["rfm_segment"]) == {"At_Risk", "Champions", "Engaged", "Occasional"}
    np.testing.assert_array_equal(mart["has_previous_month"], grouped.cumcount().gt(0))
    np.testing.assert_allclose(
        mart["data_gb_mom_change"],
        100 * grouped["data_gb"].pct_change().fillna(0),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        mart["arpu_3m_avg"],
        grouped["arpu_kzt"].rolling(3, min_periods=1).mean().to_numpy(),
    )
    np.testing.assert_array_equal(mart["support_calls_cum"], grouped["support_calls"].cumsum())
    eda = json.loads(eda_path.read_text("utf-8"))
    assert set(eda) == {"tariff_plan", "region", "rfm_segment", "month"}
    for rows in eda.values():
        assert sum(row["observations"] for row in rows) == len(raw)


def test_duckdb_windows_with_known_answer(tmp_path: Path) -> None:
    raw_path = generate_data(
        tmp_path / "raw.parquet", n_users=2, as_of=date(2026, 11, 1)
    )
    raw = pd.read_parquet(raw_path)
    one_user = raw.loc[raw["user_id"].eq(1)].iloc[[0, 1, 2, 4, 5]].copy()
    one_user["data_gb"] = [10, 20, 30, 80, 40]
    one_user["app_sessions"] = [0, 4, 2, 9, 3]
    one_user["arpu_kzt"] = [1000, 2000, 3000, 8000, 4000]
    one_user["support_calls"] = [1, 2, 3, 4, 5]
    raw = pd.concat([one_user, raw.loc[raw["user_id"].eq(2)]])
    raw.sample(frac=1, random_state=42).to_parquet(raw_path, index=False)
    mart_path = build_features(raw_path, tmp_path / "mart.parquet", tmp_path / "eda.json")
    mart = pd.read_parquet(mart_path).query("user_id == 1")
    assert mart["has_previous_month"].tolist() == [False, True, True, False, True]
    assert mart["data_gb_mom_change"].tolist() == [0, 100, 50, 0, -50]
    np.testing.assert_allclose(mart["app_sessions_mom_change"], [0, 400, -50, 0, -200 / 3])
    assert mart["data_gb_3m_avg"].tolist() == [10, 15, 20, 55, 60]
    assert mart["arpu_3m_avg"].tolist() == [1000, 1500, 2000, 5500, 6000]
    assert mart["support_calls_cum"].tolist() == [1, 3, 6, 10, 15]


def test_sql_features_ignore_outcomes_and_future_observations(tmp_path: Path) -> None:
    raw_path = generate_data(
        tmp_path / "raw.parquet", n_users=30, as_of=date(2026, 11, 1)
    )
    mart_path, eda_path = tmp_path / "mart.parquet", tmp_path / "eda.json"
    original = pd.read_parquet(build_features(raw_path, mart_path, eda_path))
    raw = pd.read_parquet(raw_path)
    cutoff = date(2026, 7, 1)
    raw.loc[raw["month"].gt(cutoff), ["data_gb", "app_sessions", "arpu_kzt", "support_calls"]] = 999
    raw["churn"] = 1 - raw["churn"]
    raw["offer_accepted"] = 1 - raw["offer_accepted"]
    raw["historical_offer"] = NO_OFFER
    raw.to_parquet(raw_path, index=False)
    changed = pd.read_parquet(build_features(raw_path, mart_path, eda_path))
    pd.testing.assert_frame_equal(
        original.loc[original["month"].le(cutoff), FEATURES],
        changed.loc[changed["month"].le(cutoff), FEATURES],
    )


def test_duplicate_subscriber_month_is_rejected(tmp_path: Path) -> None:
    raw_path = generate_data(tmp_path / "raw.parquet", n_users=2)
    raw = pd.read_parquet(raw_path)
    pd.concat([raw, raw.iloc[:1]]).to_parquet(raw_path, index=False)
    with pytest.raises(ValueError, match="unique subscriber/month-start"):
        build_features(raw_path, tmp_path / "mart.parquet", tmp_path / "eda.json")
