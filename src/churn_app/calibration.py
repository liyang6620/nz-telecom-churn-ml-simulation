"""Probability calibration independent of the persisted XGBoost artifact."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class PlattCalibrator:
    """Fit a logistic mapping on validation-set model logits."""

    estimator: LogisticRegression | None = None

    @staticmethod
    def _logits(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
        self.estimator = LogisticRegression(random_state=42)
        self.estimator.fit(self._logits(probabilities), labels)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("The probability calibrator has not been fitted.")
        return self.estimator.predict_proba(self._logits(probabilities))[:, 1]
