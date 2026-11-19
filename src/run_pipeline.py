"""Build the telecom demo end to end with one command."""

import argparse
import os
from datetime import date

from src.config import REPORT_AS_OF


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=REPORT_AS_OF,
        help=f"Data cutoff for mature labels (YYYY-MM-DD, default {REPORT_AS_OF})",
    )
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

    import pandas as pd

    from src.build_dashboard import build_dashboard
    from src.build_features import build_features
    from src.generate_data import generate_data
    from src.nbo_engine import simulate_nbo
    from src.train_churn import temporal_split, train_churn

    print("[1/5] Generating subscriber-month panel", flush=True)
    raw_path = generate_data(as_of=args.as_of)
    print("[2/5] Building DuckDB feature mart and EDA", flush=True)
    mart_path = build_features(raw_path)
    mart = pd.read_parquet(mart_path)
    train, validation, test = temporal_split(mart)
    print("[3/5] Comparing churn models", flush=True)
    train_churn(train, validation, test)
    print("[4/5] Estimating NBO and evaluating three policies", flush=True)
    simulate_nbo(pd.concat([train, validation]), test)
    print("[5/5] Rendering dashboard", flush=True)
    print(build_dashboard(mart), flush=True)


if __name__ == "__main__":
    main()
