"""Simulate pre-offer features, randomized assignment, response and next-month churn."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ASSIGNMENT_PROBABILITY,
    DATA_OFFER,
    DISCOUNT_OFFER,
    MAX_RELATIVE_CHURN_REDUCTION,
    N_MONTHS,
    N_USERS,
    NO_OFFER,
    OFFERS,
    RANDOM_SEED,
    RAW_DIR,
    ROAMING_OFFER,
    TARIFF_MARGIN_RATE,
)

REGIONS = ("Almaty", "Astana", "Shymkent", "Karaganda", "Other")
TARIFFS = tuple(TARIFF_MARGIN_RATE)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def generate_data(
    output: Path = RAW_DIR / "telecom_usage.parquet",
    *,
    n_users: int = N_USERS,
    n_months: int = N_MONTHS,
    as_of: date | None = None,
) -> Path:
    """Generate complete snapshots whose next-month labels are available as of the cutoff."""
    rng = np.random.default_rng(RANDOM_SEED)
    # Snapshot t is scored at the start of t+1. Its label matures at the start of t+2.
    last_month = pd.Period(as_of or date.today(), freq="M") - 2
    months = pd.period_range(end=last_month, periods=n_months, freq="M")
    user_id = np.arange(1, n_users + 1)
    region = rng.choice(REGIONS, n_users, p=[0.27, 0.20, 0.15, 0.12, 0.26])
    plan = rng.choice(len(TARIFFS), n_users, p=[0.31, 0.37, 0.19, 0.13])
    tariff = np.asarray(TARIFFS)[plan]
    tenure = rng.integers(1, 97, n_users)
    price_sensitive = (plan < 2) & (rng.random(n_users) < 0.58)
    traveler = rng.random(n_users) < np.where(np.isin(region, REGIONS[:2]), 0.24, 0.12)
    baseline_arpu = np.array([3200, 5350, 7850, 9300])[plan] + rng.normal(0, 750, n_users)
    baseline_data = np.array([13, 28, 61, 41])[plan] * rng.lognormal(0, 0.46, n_users)
    baseline_voice = rng.lognormal(4.9, 0.47, n_users) + traveler * 145
    baseline_app = (5 + np.sqrt(baseline_data) * 1.6) * rng.lognormal(0, 0.23, n_users)
    quality = rng.gamma(1.1, 0.30, n_users) + (region == "Other") * 0.16
    pressure = np.zeros(n_users)
    previous_data = baseline_data
    previous_app = baseline_app
    frames = []

    for index, month in enumerate(months):
        sudden_drop = rng.random(n_users) < (0.18 + 0.035 * price_sensitive)
        pressure = np.clip(
            0.24 * pressure
            + rng.gamma(0.75, 0.18, n_users)
            + sudden_drop * rng.uniform(0.62, 1.10, n_users),
            0,
            1.45,
        )
        seasonality = 1 + 0.08 * np.sin(2 * np.pi * month.month / 12)
        activity_change = np.exp(-1.65 * pressure + rng.normal(0, 0.13, n_users))
        data_gb = np.maximum(0.05, baseline_data * seasonality * activity_change).round(2)
        app = rng.poisson(baseline_app * np.exp(-1.65 * pressure))
        voice = rng.poisson(baseline_voice * np.exp(-0.31 * pressure))
        sms = rng.poisson(np.where(traveler, 8, 19) * np.exp(-0.30 * pressure))
        arpu = (baseline_arpu + rng.normal(0, 380, n_users) - 210 * pressure).clip(2500, 12000)
        arpu = arpu.round(2)
        complaints = rng.poisson(quality + 0.93 * pressure + 0.30 * (arpu >= 7500))
        calls = rng.poisson(0.26 + 0.66 * complaints + 0.45 * pressure)
        topup = (rng.gamma(2, 5, n_users) + 23 * pressure + 6 * price_sensitive).astype(int)

        data_drop = np.clip(1 - data_gb / previous_data, 0, 1)
        app_drop = np.clip(1 - app / np.maximum(previous_app, 1), 0, 1)
        quality_issue = (arpu >= 6700) & (complaints >= 2) & (calls >= 2)
        usage_risk = sigmoid(
            -4.85
            + 4.45 * pressure
            + 1.22 * (data_drop >= 0.45) * (app_drop >= 0.36)
            + 0.94 * quality_issue
            + 0.46 * (topup >= 32)
            + 0.30 * (tenure <= 6)
        )
        # Unobserved next-month shocks include competitor offers and relocation.
        # Use a separate hazard so they also affect subscribers with stable usage.
        external_risk = rng.beta(2, 48, n_users)

        offer = rng.choice(OFFERS, n_users, p=list(ASSIGNMENT_PROBABILITY.values()))
        data_fit = baseline_data >= 31
        eligible_discount = price_sensitive & (arpu >= 2900)
        acceptance = np.select(
            [offer == DATA_OFFER, offer == DISCOUNT_OFFER, offer == ROAMING_OFFER],
            [
                sigmoid(-1.65 + 2.8 * data_fit + 0.62 * pressure),
                sigmoid(-1.10 + 2.75 * eligible_discount + 0.50 * pressure),
                sigmoid(-2.25 + 3.7 * traveler + 0.60 * pressure),
            ],
        )
        relative_reduction = np.select(
            [offer == DATA_OFFER, offer == DISCOUNT_OFFER, offer == ROAMING_OFFER],
            [
                np.where(data_fit, MAX_RELATIVE_CHURN_REDUCTION, 0.15),
                np.where(eligible_discount, 0.33, 0.15),
                np.where(traveler, MAX_RELATIVE_CHURN_REDUCTION, 0.15),
            ],
        )
        accepted = (offer != NO_OFFER) & (rng.random(n_users) < acceptance)
        addressable_risk = usage_risk * (1 - accepted * relative_reduction)
        observed_risk = 1 - (1 - addressable_risk) * (1 - external_risk)
        churn = rng.random(n_users) < observed_risk
        frames.append(
            pd.DataFrame(
                {
                    "user_id": user_id,
                    "month": month.start_time.date(),
                    "region": region,
                    "tariff_plan": tariff,
                    "tenure_months": tenure + index,
                    "arpu_kzt": arpu,
                    "data_gb": data_gb,
                    "voice_min": voice,
                    "sms_count": sms,
                    "support_calls": calls,
                    "network_complaints": complaints,
                    "days_since_last_topup": topup,
                    "app_sessions": app,
                    "historical_offer": offer,
                    "offer_accepted": accepted.astype("int8"),
                    "churn": churn.astype("int8"),
                }
            )
        )
        previous_data, previous_app = data_gb, app

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(output, index=False)
    return output
