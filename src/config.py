"""Pipeline paths and shared modelling and business assumptions."""

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
DOCS_DIR = ROOT / "docs"
SQL_DIR = ROOT / "sql"

RANDOM_SEED = 42
N_USERS = 50_000
N_MONTHS = 6
TOP_FRACTION = 0.10
REPORT_AS_OF = date(2026, 11, 19)

NO_OFFER = "No_Offer"
DATA_OFFER = "Unlimited_Data_30d"
DISCOUNT_OFFER = "Discount_20pct"
ROAMING_OFFER = "Roaming_Pack"
OFFERS = (NO_OFFER, DATA_OFFER, DISCOUNT_OFFER, ROAMING_OFFER)
ASSIGNMENT_PROBABILITY = dict(zip(OFFERS, (0.50, 1 / 6, 1 / 6, 1 / 6), strict=True))
OFFER_COST_KZT = {NO_OFFER: 0, DATA_OFFER: 450, DISCOUNT_OFFER: 0, ROAMING_OFFER: 600}
OFFER_COST_ARPU_FRACTION = {DISCOUNT_OFFER: 0.20}
CONTACT_COST_KZT = 75
MAX_RELATIVE_CHURN_REDUCTION = 0.35
RETENTION_MONTHS = 3
TARIFF_MARGIN_RATE = {"Basic": 0.60, "Smart": 0.65, "Unlimited": 0.70, "Family": 0.65}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
