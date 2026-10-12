"""Deterministic decision support built on top of churn probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

RISK_ORDER = ["Critical", "High", "Medium", "Low"]


def _risk_tier(score: pd.Series) -> pd.Categorical:
    labels = np.select(
        [score.ge(0.75), score.ge(0.50), score.ge(0.25)],
        ["Critical", "High", "Medium"],
        default="Low",
    )
    return pd.Categorical(labels, categories=RISK_ORDER, ordered=True)


def _recommended_action(row: pd.Series) -> str:
    satisfaction = row.get("Customer Satisfaction")
    service_requests = row.get("Total Customer Svc Requests", 0)
    issues = row.get("Product/Service Issues Reported", 0)
    if (pd.notna(satisfaction) and satisfaction <= 2) or service_requests >= 3 or issues >= 2:
        return "Service recovery call"
    if str(row.get("Contract", "")).lower().startswith("month"):
        return "Review contract offer"
    if row.get("Online Security") == "No" or row.get("Premium Tech Support") == "No":
        return "Review service bundle"
    return "Loyalty check-in"


def build_retention_plan(scored_df: pd.DataFrame, contact_capacity: int) -> pd.DataFrame:
    """Rank customers and add transparent operational decision fields."""
    result = scored_df.copy()
    result["risk_tier"] = _risk_tier(result["churn_risk_score"])
    result["monthly_revenue_at_risk"] = (
        result["Monthly Charge"].fillna(0) * result["churn_risk_score"]
    ).round(2)
    result["recommended_action"] = result.apply(_recommended_action, axis=1)
    result = result.sort_values(
        ["churn_risk_score", "monthly_revenue_at_risk"], ascending=False
    ).reset_index(drop=True)
    result["priority_rank"] = np.arange(1, len(result) + 1)
    capacity = max(0, min(int(contact_capacity), len(result)))
    result["contact_this_cycle"] = result["priority_rank"].le(capacity)
    return result
