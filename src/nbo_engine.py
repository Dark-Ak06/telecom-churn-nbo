"""Estimate offer responses and evaluate retention policies on randomized logs."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    ASSIGNMENT_PROBABILITY,
    CONTACT_COST_KZT,
    DISCOUNT_OFFER,
    MAX_RELATIVE_CHURN_REDUCTION,
    NO_OFFER,
    OFFER_COST_ARPU_FRACTION,
    OFFER_COST_KZT,
    OFFERS,
    RETENTION_MONTHS,
    TARIFF_MARGIN_RATE,
    write_json,
)
from src.train_churn import FEATURES, boosting_model, require_mature_history


def select_offers(profits: pd.DataFrame) -> pd.Series:
    active = profits.loc[:, list(OFFERS[1:])]
    return active.idxmax(axis=1).where(active.max(axis=1) > 0, NO_OFFER)


def predict_offer_outcomes(
    history: pd.DataFrame,
    subscribers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_mature_history(history, subscribers)
    features = subscribers[FEATURES]
    churn_risk = pd.DataFrame(index=subscribers.index)
    response = pd.DataFrame({NO_OFFER: 0.0}, index=subscribers.index)
    for offer in OFFERS:
        arm = history.loc[history["historical_offer"].eq(offer)]
        churn = boosting_model(max_leaf_nodes=15, max_iter=100).fit(arm[FEATURES], arm["churn"])
        churn_risk[offer] = churn.predict_proba(features)[:, 1]
        if offer != NO_OFFER:
            acceptance = boosting_model(max_leaf_nodes=15, max_iter=75).fit(
                arm[FEATURES],
                arm["offer_accepted"],
            )
            response[offer] = acceptance.predict_proba(features)[:, 1]
    return churn_risk, response


def offer_profits(
    churn_risk: pd.DataFrame,
    response: pd.DataFrame,
    retention_value: pd.Series,
    redemption_costs: pd.DataFrame,
) -> pd.DataFrame:
    active_offers = list(OFFERS[1:])
    acceptance = response[active_offers]
    assignment_effect = churn_risk[active_offers].rsub(churn_risk[NO_OFFER], axis=0)
    # Keep adverse estimates and cap optimistic ITT gains for the decision rule.
    benefit_cap = acceptance.mul(churn_risk[NO_OFFER] * MAX_RELATIVE_CHURN_REDUCTION, axis=0)
    expected_effect = np.minimum(assignment_effect, benefit_cap)
    return (
        expected_effect.mul(retention_value, axis=0)
        - acceptance * redemption_costs[active_offers]
        - CONTACT_COST_KZT
    )


def simulate_nbo(
    history: pd.DataFrame,
    test: pd.DataFrame,
    output: Path = ARTIFACTS_DIR / "nbo_simulation.json",
) -> dict:
    """Estimate policy values with cross-period outcome models and known assignment odds."""
    if test["month"].nunique() != 1:
        raise ValueError("NBO policy evaluation requires exactly one test month")
    churn_risk, response = predict_offer_outcomes(history, test)
    arpu = test["arpu_kzt"]
    gross_value = arpu * RETENTION_MONTHS
    retention_value = gross_value * test["tariff_plan"].map(TARIFF_MARGIN_RATE)
    redemption_costs = pd.DataFrame(
        {
            offer: OFFER_COST_KZT[offer] + arpu * OFFER_COST_ARPU_FRACTION.get(offer, 0.0)
            for offer in OFFERS
        }
    )
    policies = {
        "do_nothing": pd.Series(NO_OFFER, index=test.index),
        "mass_discount": pd.Series(DISCOUNT_OFFER, index=test.index),
        "ml_nbo": select_offers(
            offer_profits(churn_risk, response, retention_value, redemption_costs)
        ),
    }
    logged_action = test["historical_offer"]
    propensity = logged_action.map(ASSIGNMENT_PROBABILITY).to_numpy()
    offer_codes = {offer: column for column, offer in enumerate(OFFERS)}
    rows = np.arange(len(test))
    modeled_risks = churn_risk.loc[:, list(OFFERS)].to_numpy()
    modeled_response = response.loc[:, list(OFFERS)].to_numpy()
    costs_by_offer = redemption_costs.loc[:, list(OFFERS)].to_numpy()
    logged_columns = logged_action.map(offer_codes).to_numpy()
    logged_risk = modeled_risks[rows, logged_columns]
    logged_response = modeled_response[rows, logged_columns]
    churn_correction = (test["churn"].to_numpy() - logged_risk) / propensity
    response_correction = (test["offer_accepted"].to_numpy() - logged_response) / propensity
    control_risk = (
        modeled_risks[:, offer_codes[NO_OFFER]]
        + churn_correction * logged_action.eq(NO_OFFER).to_numpy()
    )

    strategies = {}
    for policy, recommended in policies.items():
        columns = recommended.map(offer_codes).to_numpy()
        matches = recommended.eq(logged_action).to_numpy()
        policy_risk = modeled_risks[rows, columns] + churn_correction * matches
        policy_response = modeled_response[rows, columns] + response_correction * matches
        saved_risk = control_risk - policy_risk
        contacted = recommended.ne(NO_OFFER)
        contact_cost = CONTACT_COST_KZT * int(contacted.sum())
        redemption_cost = policy_response * costs_by_offer[rows, columns]
        cost = CONTACT_COST_KZT * contacted.to_numpy() + redemption_cost
        net_by_user = saved_risk * retention_value - cost
        campaign_cost = float(cost.sum())
        net = float(net_by_user.sum())
        se_total = float(net_by_user.std(ddof=1) * np.sqrt(len(test)))
        lost_margin = float((policy_risk * retention_value).sum())
        strategies[policy] = {
            "churn_rate": float(policy_risk.mean()),
            "churn_reduction_pp": float(100 * saved_risk.mean()),
            "expected_retained_subscribers": float(saved_risk.sum()),
            "lost_revenue_kzt": float((policy_risk * gross_value).sum()),
            "contact_cost_kzt": contact_cost,
            "redemption_cost_kzt": float(redemption_cost.sum()),
            "campaign_cost_kzt": campaign_cost,
            "net_retained_margin_kzt": float(retention_value.sum() - lost_margin - campaign_cost),
            "net_incremental_margin_kzt": net,
            "net_incremental_margin_ci95_kzt": [net - 1.96 * se_total, net + 1.96 * se_total],
            "roi_pct": 100 * net / campaign_cost if campaign_cost else None,
            "targeted_subscribers": int(contacted.sum()),
        }

    by_segment = pd.crosstab(test["rfm_segment"], policies["ml_nbo"]).reindex(
        columns=list(OFFERS),
        fill_value=0,
    )
    results = {
        "method": "doubly_robust_randomized_holdout",
        "test_month": str(test["month"].iloc[0]),
        "outcome_month": str(pd.Period(test["month"].iloc[0], freq="M") + 1),
        "subscribers": len(test),
        "horizon_months": RETENTION_MONTHS,
        "tariff_margin_rate": TARIFF_MARGIN_RATE,
        "contact_cost_per_assignment_kzt": CONTACT_COST_KZT,
        "decision_relative_churn_reduction_cap": MAX_RELATIVE_CHURN_REDUCTION,
        "assignment_probability": ASSIGNMENT_PROBABILITY,
        "strategies": strategies,
        "ml_offer_counts_by_rfm": by_segment.reset_index().to_dict(orient="records"),
    }
    write_json(output, results)
    return results
