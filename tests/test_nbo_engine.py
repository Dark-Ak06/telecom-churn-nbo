import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.build_features import build_features
from src.config import (
    CONTACT_COST_KZT,
    DATA_OFFER,
    DISCOUNT_OFFER,
    NO_OFFER,
    OFFERS,
    ROAMING_OFFER,
    TARIFF_MARGIN_RATE,
)
from src.generate_data import generate_data
from src.nbo_engine import offer_profits, select_offers, simulate_nbo
from src.train_churn import temporal_split


def test_offer_rule_matches_profit_and_leaves_zero_or_negative_untargeted() -> None:
    churn = pd.DataFrame(
        {
            NO_OFFER: [0.875, 0.875, 0.875, 0.875],
            DATA_OFFER: [0.25, 0.75, 1.0, 0.25],
            DISCOUNT_OFFER: [0.875, 0.875, 0.875, 0.75],
            ROAMING_OFFER: [0.875, 0.875, 0.875, 0.875],
        }
    )
    response = pd.DataFrame(0.5, index=churn.index, columns=list(OFFERS))
    response.loc[3, DATA_OFFER] = 0.0
    costs = pd.DataFrame(
        {NO_OFFER: 0, DATA_OFFER: 350, DISCOUNT_OFFER: 200, ROAMING_OFFER: 450},
        index=churn.index,
    )
    profit = offer_profits(churn, response, pd.Series(2000, index=churn.index), costs)
    assert profit.loc[0, DATA_OFFER] == pytest.approx(56.25)
    assert profit.loc[1, DATA_OFFER] == 0
    assert profit.loc[2, DATA_OFFER] == -500
    assert profit.loc[3, DATA_OFFER] == -CONTACT_COST_KZT
    assert select_offers(profit).tolist() == [DATA_OFFER, NO_OFFER, NO_OFFER, DISCOUNT_OFFER]


def test_randomized_holdout_nbo_outperforms_mass_discount(tmp_path: Path) -> None:
    raw = generate_data(tmp_path / "raw.parquet", n_users=18_000, as_of=date(2026, 11, 1))
    mart = build_features(raw, tmp_path / "mart.parquet", tmp_path / "eda.json")
    train, validation, test = temporal_split(pd.read_parquet(mart))
    output = tmp_path / "simulation.json"
    result = simulate_nbo(pd.concat([train, validation]), test, output)
    control, mass, targeted = (
        result["strategies"][key] for key in ("do_nothing", "mass_discount", "ml_nbo")
    )
    assert control["campaign_cost_kzt"] == control["net_incremental_margin_kzt"] == 0
    assert control["roi_pct"] is None
    assert mass["contact_cost_kzt"] == CONTACT_COST_KZT * len(test)
    assert 0 < mass["redemption_cost_kzt"] < 0.2 * test["arpu_kzt"].sum()
    assert 0 < targeted["targeted_subscribers"] < len(test)
    assert targeted["net_incremental_margin_kzt"] > max(0, mass["net_incremental_margin_kzt"])
    assert targeted["roi_pct"] > 0
    for policy in (mass, targeted):
        assert policy["campaign_cost_kzt"] == pytest.approx(
            policy["contact_cost_kzt"] + policy["redemption_cost_kzt"]
        )
        assert policy["net_incremental_margin_kzt"] == pytest.approx(
            policy["net_retained_margin_kzt"] - control["net_retained_margin_kzt"]
        )
        assert policy["roi_pct"] == pytest.approx(
            100 * policy["net_incremental_margin_kzt"] / policy["campaign_cost_kzt"]
        )
    assigned = sum(sum(row[offer] for offer in OFFERS) for row in result["ml_offer_counts_by_rfm"])
    assert assigned == len(test)
    assert output.is_file()
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="labels must be mature"):
        simulate_nbo(test, test, output)


def test_policy_charges_redemptions_regardless_of_churn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One treated row per offer among six rows makes the randomization weights transparent.
    test = pd.DataFrame(
        {
            "month": date(2026, 9, 1),
            "arpu_kzt": 5000,
            "tariff_plan": "Smart",
            "rfm_segment": "Occasional",
            "historical_offer": [NO_OFFER, NO_OFFER, NO_OFFER, *OFFERS[1:]],
            "offer_accepted": [0, 0, 0, 0, 1, 0],
            "churn": 0,
        }
    )
    history = test.assign(month=date(2026, 8, 1))
    risks = pd.DataFrame(0.25, index=test.index, columns=list(OFFERS))
    responses = pd.DataFrame(0.50, index=test.index, columns=list(OFFERS))
    responses[NO_OFFER] = 0.0

    def predict(training_rows: pd.DataFrame, scoring_rows: pd.DataFrame) -> tuple:
        assert training_rows["month"].max() < scoring_rows["month"].min()
        return risks, responses

    monkeypatch.setattr("src.nbo_engine.predict_offer_outcomes", predict)
    kept = simulate_nbo(history, test, tmp_path / "kept.json")["strategies"]["mass_discount"]
    test.loc[test["offer_accepted"].eq(1), "churn"] = 1
    churned = simulate_nbo(history, test, tmp_path / "churned.json")["strategies"]["mass_discount"]
    assert kept["campaign_cost_kzt"] == churned["campaign_cost_kzt"] == 6 * (75 + 1000)
    assert kept["net_incremental_margin_kzt"] - churned["net_incremental_margin_kzt"] == (
        6 * 5000 * 3 * TARIFF_MARGIN_RATE["Smart"]
    )
