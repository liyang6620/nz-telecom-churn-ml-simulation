"""Leakage-aware model selection, calibration, evaluation, and artifact generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .calibration import PlattCalibrator
from .config import (
    CALIBRATOR_PATH,
    CUSTOMER_ID_COLUMN,
    LEAKAGE_COLUMNS,
    MODEL_FEATURES,
    MODEL_PATH,
    PROJECT_ROOT,
    RAW_FEATURES_PATH,
    SCHEMA_PATH,
)
from .scoring import prepare_features

RANDOM_STATE = 42
TARGET = "Churn Value"
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn_data.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f2: float
    balanced_accuracy: float
    specificity: float
    false_positives: int
    false_negatives: int


def _models() -> dict[str, object]:
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=RANDOM_STATE),
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost (shallow)": XGBClassifier(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.04,
            min_child_weight=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost (regularised)": XGBClassifier(
            n_estimators=450,
            max_depth=2,
            learning_rate=0.035,
            min_child_weight=6,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=4.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def _model_comparison(
    x_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, str, object]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float | str]] = []
    models = _models()
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "brier": "neg_brier_score",
    }
    for name, model in models.items():
        scores = cross_validate(model, x_train, y_train, cv=folds, scoring=scoring, n_jobs=1)
        rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
                "cv_roc_auc_std": scores["test_roc_auc"].std(ddof=1),
                "cv_pr_auc_mean": scores["test_pr_auc"].mean(),
                "cv_pr_auc_std": scores["test_pr_auc"].std(ddof=1),
                "cv_brier_mean": -scores["test_brier"].mean(),
                "cv_brier_std": scores["test_brier"].std(ddof=1),
            }
        )
    comparison = pd.DataFrame(rows).sort_values(
        ["cv_roc_auc_mean", "cv_pr_auc_mean"], ascending=False
    )
    selected_name = str(comparison.iloc[0]["model"])
    selected = models[selected_name]
    if not isinstance(selected, XGBClassifier):
        raise RuntimeError(
            f"{selected_name} won cross-validation; update persistence before replacing XGBoost."
        )
    return comparison, selected_name, selected


def _choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Minimise a documented 5:1 false-negative/false-positive validation cost."""
    candidates = np.unique(np.r_[np.linspace(0.05, 0.8, 151), probabilities])
    best = min(
        candidates,
        key=lambda threshold: (
            5 * np.sum((probabilities < threshold) & (y_true.to_numpy() == 1))
            + np.sum((probabilities >= threshold) & (y_true.to_numpy() == 0)),
            threshold,
        ),
    )
    return float(best)


def _threshold_metrics(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> ThresholdMetrics:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return ThresholdMetrics(
        threshold=threshold,
        precision=precision_score(y_true, predictions, zero_division=0),
        recall=recall_score(y_true, predictions, zero_division=0),
        f2=fbeta_score(y_true, predictions, beta=2, zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y_true, predictions),
        specificity=tn / (tn + fp),
        false_positives=int(fp),
        false_negatives=int(fn),
    )


def _bootstrap_interval(
    y_true: pd.Series, probabilities: np.ndarray, metric, repeats: int = 1_000
) -> list[float]:
    rng = np.random.default_rng(RANDOM_STATE)
    labels = y_true.to_numpy()
    values: list[float] = []
    for _ in range(repeats):
        indices = rng.integers(0, len(labels), len(labels))
        if np.unique(labels[indices]).size == 2:
            values.append(float(metric(labels[indices], probabilities[indices])))
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _top_shap_features(
    model: XGBClassifier, x_test: pd.DataFrame, limit: int = 15
) -> pd.DataFrame:
    matrix = xgb.DMatrix(x_test, feature_names=list(x_test.columns))
    contributions = model.get_booster().predict(matrix, pred_contribs=True)[:, :-1]
    return (
        pd.DataFrame(
            {
                "feature": x_test.columns,
                "mean_abs_shap": np.abs(contributions).mean(axis=0),
                "mean_shap": contributions.mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )


def _save_figures(
    comparison: pd.DataFrame,
    y_test: pd.Series,
    raw_test: np.ndarray,
    calibrated_test: np.ndarray,
    threshold: float,
    top_features: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    ordered = comparison.sort_values("cv_roc_auc_mean")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(
        ordered["model"],
        ordered["cv_roc_auc_mean"],
        xerr=ordered["cv_roc_auc_std"],
        color="#16754a",
        alpha=0.9,
    )
    ax.set(
        xlabel="5-fold ROC-AUC (mean +/- 1 SD)",
        xlim=(0.5, 1.0),
        title="Model selection on training data only",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model-comparison.png", dpi=180)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, calibrated_test)
    precision, recall, _ = precision_recall_curve(y_test, calibrated_test)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].plot(fpr, tpr, color="#16754a", lw=2)
    axes[0].plot([0, 1], [0, 1], "--", color="#89928d")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[1].plot(recall, precision, color="#c05a32", lw=2)
    axes[1].axhline(y_test.mean(), ls="--", color="#89928d")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "test-curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for label, values, colour in (
        ("Raw XGBoost", raw_test, "#89928d"),
        ("Platt calibrated", calibrated_test, "#16754a"),
    ):
        observed, predicted = calibration_curve(y_test, values, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=label, color=colour)
    ax.plot([0, 1], [0, 1], "--", color="#c05a32", label="Perfect calibration")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed churn rate",
        title="Calibration on untouched test data",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration.png", dpi=180)
    plt.close(fig)

    predictions = (calibrated_test >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=["Stayed", "Churned"], cmap="Greens", ax=ax
    )
    ax.set_title(f"Test confusion matrix at threshold {threshold:.3f}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion-matrix.png", dpi=180)
    plt.close(fig)

    ordered_features = top_features.sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(ordered_features["feature"], ordered_features["mean_abs_shap"], color="#16754a")
    ax.set(
        xlabel="Mean absolute SHAP value (log-odds)",
        title="Global feature influence on test predictions",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature-importance.png", dpi=180)
    plt.close(fig)


def train_and_evaluate() -> dict[str, object]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    raw_features = tuple(MODEL_FEATURES)
    labels = raw[TARGET].astype(int)

    train_indices, remainder_indices = train_test_split(
        np.arange(len(raw)), test_size=0.4, stratify=labels, random_state=RANDOM_STATE
    )
    validation_indices, test_indices = train_test_split(
        remainder_indices,
        test_size=0.5,
        stratify=labels.iloc[remainder_indices],
        random_state=RANDOM_STATE,
    )
    x_train = prepare_features(raw.iloc[train_indices], raw_features)
    training_columns = list(x_train.columns)
    x_validation = prepare_features(raw.iloc[validation_indices], raw_features).reindex(
        columns=training_columns, fill_value=0
    )
    x_test = prepare_features(raw.iloc[test_indices], raw_features).reindex(
        columns=training_columns, fill_value=0
    )
    y_train = labels.iloc[train_indices]
    y_validation = labels.iloc[validation_indices]
    y_test = labels.iloc[test_indices]

    comparison, selected_name, model = _model_comparison(x_train, y_train)
    model.fit(x_train, y_train)
    raw_validation = model.predict_proba(x_validation)[:, 1]
    calibrator = PlattCalibrator().fit(raw_validation, y_validation.to_numpy())
    calibrated_validation = calibrator.predict(raw_validation)
    threshold = _choose_threshold(y_validation, calibrated_validation)

    raw_test = model.predict_proba(x_test)[:, 1]
    calibrated_test = calibrator.predict(raw_test)
    threshold_metrics = _threshold_metrics(y_test, calibrated_test, threshold)
    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, calibrated_test)),
        "roc_auc_95_ci": _bootstrap_interval(y_test, calibrated_test, roc_auc_score),
        "pr_auc": float(average_precision_score(y_test, calibrated_test)),
        "pr_auc_95_ci": _bootstrap_interval(y_test, calibrated_test, average_precision_score),
        "brier_raw": float(brier_score_loss(y_test, raw_test)),
        "brier_calibrated": float(brier_score_loss(y_test, calibrated_test)),
        "log_loss": float(log_loss(y_test, calibrated_test)),
        **asdict(threshold_metrics),
    }
    top_features = _top_shap_features(model, x_test)

    model.save_model(MODEL_PATH)
    CALIBRATOR_PATH.write_text(
        json.dumps(calibrator.to_dict(), indent=2), encoding="utf-8"
    )
    joblib.dump(training_columns, SCHEMA_PATH)
    joblib.dump(list(raw_features), RAW_FEATURES_PATH)
    comparison.to_csv(TABLES_DIR / "model-comparison.csv", index=False)
    top_features.to_csv(TABLES_DIR / "top-features.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(TABLES_DIR / "test-metrics.csv", index=False)

    split_assignments = pd.DataFrame(
        {
            CUSTOMER_ID_COLUMN: raw[CUSTOMER_ID_COLUMN],
            "split": "train",
        }
    )
    split_assignments.loc[validation_indices, "split"] = "validation"
    split_assignments.loc[test_indices, "split"] = "test"
    split_assignments.to_csv(ARTIFACTS_DIR / "split_assignments.csv", index=False)

    metadata: dict[str, object] = {
        "model_version": "2.0.0",
        "random_state": RANDOM_STATE,
        "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "dataset_rows": len(raw),
        "target_rate": float(labels.mean()),
        "split": {
            "train_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "test_rows": len(test_indices),
        },
        "selection": {
            "method": "5-fold stratified CV on the training split",
            "primary_metric": "ROC-AUC",
            "selected_model": selected_name,
            "candidates": comparison.to_dict(orient="records"),
        },
        "calibration": "Platt scaling fitted on the validation split",
        "threshold_policy": "Validation cost: false negative = 5, false positive = 1",
        "test_metrics": test_metrics,
        "top_features": top_features.to_dict(orient="records"),
        "features": list(raw_features),
        "excluded_as_direct_leakage": list(LEAKAGE_COLUMNS),
        "excluded_as_timing_risk": ["Customer Satisfaction"],
    }
    (ARTIFACTS_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _save_figures(
        comparison, y_test, raw_test, calibrated_test, threshold, top_features
    )
    return metadata


if __name__ == "__main__":
    result = train_and_evaluate()
    print(json.dumps(result["test_metrics"], indent=2))
