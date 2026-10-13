import json

import numpy as np
import pandas as pd

from churn_app.calibration import PlattCalibrator
from churn_app.config import MODEL_METADATA_PATH, PROJECT_ROOT


def test_calibrator_returns_bounded_monotonic_probabilities():
    calibrator = PlattCalibrator().fit(
        probabilities=np.array([0.05, 0.2, 0.7, 0.9]),
        labels=np.array([0, 0, 1, 1]),
    )
    calibrated = calibrator.predict(np.array([0.1, 0.5, 0.8]))

    assert ((calibrated > 0) & (calibrated < 1)).all()
    assert (calibrated[1:] > calibrated[:-1]).all()


def test_published_metadata_has_isolated_evaluation_contract():
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))

    assert metadata["split"] == {
        "train_rows": 4225,
        "validation_rows": 1409,
        "test_rows": 1409,
    }
    assert metadata["selection"]["selected_model"] == "XGBoost (shallow)"
    assert metadata["test_metrics"]["roc_auc"] > 0.9
    assert metadata["test_metrics"]["pr_auc"] > 0.85
    assert metadata["test_metrics"]["brier_calibrated"] <= metadata["test_metrics"]["brier_raw"]
    assert metadata["excluded_as_timing_risk"] == ["Customer Satisfaction"]


def test_split_manifest_assigns_each_customer_once():
    source = pd.read_csv(PROJECT_ROOT / "data" / "telco_churn_data.csv")
    manifest = pd.read_csv(PROJECT_ROOT / "artifacts" / "split_assignments.csv")

    assert manifest["Customer ID"].is_unique
    assert set(manifest["Customer ID"]) == set(source["Customer ID"])
    assert manifest["split"].value_counts().to_dict() == {
        "train": 4225,
        "validation": 1409,
        "test": 1409,
    }
