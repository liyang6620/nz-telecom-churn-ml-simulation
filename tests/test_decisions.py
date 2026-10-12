import pandas as pd

from churn_app.decisions import build_retention_plan


def test_retention_plan_ranks_and_respects_capacity():
    source = pd.DataFrame(
        {
            "Customer ID": ["A", "B", "C"],
            "churn_risk_score": [0.2, 0.9, 0.6],
            "Monthly Charge": [50, 100, 80],
            "Customer Satisfaction": [5, 1, 4],
            "Total Customer Svc Requests": [0, 4, 0],
            "Product/Service Issues Reported": [0, 2, 0],
            "Contract": ["Two Year", "Month-to-Month", "One Year"],
            "Online Security": ["Yes", "No", "No"],
            "Premium Tech Support": ["Yes", "No", "No"],
        }
    )

    result = build_retention_plan(source, contact_capacity=2)

    assert result["Customer ID"].tolist() == ["B", "C", "A"]
    assert result["risk_tier"].astype(str).tolist() == ["Critical", "High", "Low"]
    assert result["contact_this_cycle"].tolist() == [True, True, False]
    assert result.loc[0, "recommended_action"] == "Service recovery call"


def test_capacity_is_bounded_by_customer_count():
    source = pd.DataFrame(
        {
            "Customer ID": ["A"],
            "churn_risk_score": [0.4],
            "Monthly Charge": [50],
            "Customer Satisfaction": [3],
            "Total Customer Svc Requests": [0],
            "Product/Service Issues Reported": [0],
            "Contract": ["Month-to-Month"],
            "Online Security": ["Yes"],
            "Premium Tech Support": ["Yes"],
        }
    )
    result = build_retention_plan(source, contact_capacity=10)
    assert result["contact_this_cycle"].tolist() == [True]
