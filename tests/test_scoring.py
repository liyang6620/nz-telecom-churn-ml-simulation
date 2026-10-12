import numpy as np
import pandas as pd
import pytest

from churn_app.config import PROJECT_ROOT
from churn_app.scoring import DataValidationError, load_artifacts, score_customers


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


def test_demo_scores_match_regression_baseline(artifacts):
    source = pd.read_csv(PROJECT_ROOT / "data" / "telco_test_sample.csv")
    expected = pd.read_csv(PROJECT_ROOT / "data" / "scored_customers.csv")

    actual, _ = score_customers(source, artifacts)
    joined = actual[["Customer ID", "churn_risk_score"]].merge(
        expected[["Customer ID", "churn_risk_score"]],
        on="Customer ID",
        suffixes=("_actual", "_expected"),
    )

    assert len(joined) == len(source)
    np.testing.assert_allclose(
        joined["churn_risk_score_actual"], joined["churn_risk_score_expected"], rtol=1e-6
    )


def test_missing_required_column_is_actionable(artifacts):
    source = pd.read_csv(PROJECT_ROOT / "data" / "telco_test_sample.csv").drop(
        columns=["Monthly Charge"]
    )
    with pytest.raises(DataValidationError, match="Missing required columns: Monthly Charge"):
        score_customers(source, artifacts)


def test_duplicate_customer_ids_are_rejected(artifacts):
    source = pd.read_csv(PROJECT_ROOT / "data" / "telco_test_sample.csv")
    source.loc[1, "Customer ID"] = source.loc[0, "Customer ID"]
    with pytest.raises(DataValidationError, match="Customer ID must be unique"):
        score_customers(source, artifacts)


def test_invalid_numeric_value_is_rejected(artifacts):
    source = pd.read_csv(PROJECT_ROOT / "data" / "telco_test_sample.csv")
    source["Monthly Charge"] = source["Monthly Charge"].astype("object")
    source.loc[0, "Monthly Charge"] = "not-a-number"
    with pytest.raises(DataValidationError, match="Monthly Charge contains non-numeric values"):
        score_customers(source, artifacts)


def test_score_is_independent_of_upload_batch(artifacts):
    source = pd.read_csv(PROJECT_ROOT / "data" / "telco_test_sample.csv")
    batch_scores, _ = score_customers(source, artifacts)
    single_score, _ = score_customers(source.iloc[[0]], artifacts)

    assert single_score.loc[single_score.index[0], "churn_risk_score"] == pytest.approx(
        batch_scores.loc[batch_scores.index[0], "churn_risk_score"]
    )


def test_unsupported_category_is_rejected(artifacts):
    source = pd.read_csv(PROJECT_ROOT / "data" / "telco_test_sample.csv")
    source.loc[0, "Contract"] = "Weekly"
    with pytest.raises(DataValidationError, match="Contract contains unsupported values"):
        score_customers(source, artifacts)
