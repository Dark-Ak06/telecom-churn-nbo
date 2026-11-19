"""Materialize the DuckDB feature mart and export aggregate EDA."""

from pathlib import Path

import duckdb

from src.config import ARTIFACTS_DIR, PROCESSED_DIR, RAW_DIR, SQL_DIR, write_json


def build_features(
    raw_path: Path = RAW_DIR / "telecom_usage.parquet",
    mart_path: Path = PROCESSED_DIR / "feature_mart.parquet",
    eda_path: Path = ARTIFACTS_DIR / "eda_summary.json",
) -> Path:
    mart_path.parent.mkdir(parents=True, exist_ok=True)
    feature_sql = (SQL_DIR / "02_feature_mart.sql").read_text("utf-8").rstrip().removesuffix(";")
    with duckdb.connect(config={"threads": 4}) as connection:
        invalid_key = connection.execute(
            """
            SELECT user_id, month FROM read_parquet($raw_path)
            GROUP BY user_id, month
            HAVING count(*) > 1 OR user_id IS NULL OR month IS NULL
                OR month <> date_trunc('month', month)
            LIMIT 1
            """,
            {"raw_path": str(raw_path)},
        ).fetchone()
        if invalid_key is not None:
            raise ValueError(f"Expected unique subscriber/month-start keys, found {invalid_key}")
        connection.execute(
            f"CREATE TEMP TABLE feature_mart AS {feature_sql}", {"raw_path": str(raw_path)}
        )
        connection.table("feature_mart").order("month, user_id").write_parquet(str(mart_path))
        cursor = connection.execute((SQL_DIR / "01_eda_metrics.sql").read_text("utf-8"))
        columns = [column[0] for column in cursor.description]
        summary = {}
        for dimension, *values in cursor.fetchall():
            summary.setdefault(dimension, []).append(dict(zip(columns[1:], values, strict=True)))
    write_json(eda_path, summary)
    return mart_path
