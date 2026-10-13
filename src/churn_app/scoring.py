"""Validated feature preparation and model scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .config import (
    BASELINE_DUMMY_COLUMNS,
    CALIBRATOR_PATH,
    CATEGORICAL_VALUES,
    CUSTOMER_ID_COLUMN,
    LEAKAGE_COLUMNS,
    MAX_ROWS,
    MODEL_PATH,
    NUMERIC_COLUMNS,
    RAW_FEATURES_PATH,
    SCHEMA_PATH,
    TENURE_BINS,
    TENURE_COLUMN,
    TENURE_GROUP_COLUMN,
    YES_NO_COLUMNS,
)


class DataValidationError(ValueError):
    """Raised when a customer snapshot cannot be scored safely."""


@dataclass(frozen=True)
class ModelArtifacts:
    model: object
    calibrator: object
    training_columns: pd.Index
    raw_feature_list: tuple[str, ...]


def load_artifacts(
    model_path: Path = MODEL_PATH,
    calibrator_path: Path = CALIBRATOR_PATH,
    schema_path: Path = SCHEMA_PATH,
    raw_features_path: Path = RAW_FEATURES_PATH,
) -> ModelArtifacts:
    """Load the immutable model bundle used by both the app and tests."""
    model = XGBClassifier()
    model.load_model(model_path)
    return ModelArtifacts(
        model=model,
        calibrator=joblib.load(calibrator_path),
        training_columns=pd.Index(joblib.load(schema_path)),
        raw_feature_list=tuple(joblib.load(raw_features_path)),
    )


def validate_customer_data(raw_df: pd.DataFrame, raw_feature_list: tuple[str, ...]) -> pd.DataFrame:
    """Validate and normalise a customer snapshot before feature preparation."""
    if raw_df.empty:
        raise DataValidationError("The CSV contains no customer rows.")
    if len(raw_df) > MAX_ROWS:
        raise DataValidationError(
            f"The CSV has {len(raw_df):,} rows; the demo limit is {MAX_ROWS:,}."
        )
    if raw_df.columns.duplicated().any():
        duplicate_columns = raw_df.columns[raw_df.columns.duplicated()].tolist()
        raise DataValidationError(f"Duplicate column names: {', '.join(duplicate_columns)}.")

    model_inputs = [column for column in raw_feature_list if column != TENURE_GROUP_COLUMN]
    required = [CUSTOMER_ID_COLUMN, *model_inputs]
    missing = [column for column in required if column not in raw_df.columns]
    if missing:
        preview = ", ".join(missing[:6])
        suffix = "..." if len(missing) > 6 else ""
        raise DataValidationError(f"Missing required columns: {preview}{suffix}")

    clean = raw_df.copy()
    clean[CUSTOMER_ID_COLUMN] = clean[CUSTOMER_ID_COLUMN].astype("string").str.strip()
    if clean[CUSTOMER_ID_COLUMN].isna().any() or clean[CUSTOMER_ID_COLUMN].eq("").any():
        raise DataValidationError("Customer ID contains blank values.")
    duplicates = clean.loc[clean[CUSTOMER_ID_COLUMN].duplicated(), CUSTOMER_ID_COLUMN].unique()
    if len(duplicates):
        preview = ", ".join(map(str, duplicates[:5]))
        raise DataValidationError(f"Customer ID must be unique. Duplicates include: {preview}.")

    for column in (name for name in NUMERIC_COLUMNS if name in model_inputs):
        original = clean[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            examples = ", ".join(map(str, original[invalid].astype(str).unique()[:3]))
            raise DataValidationError(f"{column} contains non-numeric values: {examples}.")
        clean[column] = converted

    if clean[TENURE_COLUMN].isna().any():
        raise DataValidationError(f"{TENURE_COLUMN} cannot be blank.")
    if ~clean[TENURE_COLUMN].between(TENURE_BINS[0], TENURE_BINS[-1]).all():
        raise DataValidationError(f"{TENURE_COLUMN} must be between 0 and 72.")
    for column in ("Monthly Charge", "Total Regular Charges"):
        if clean[column].isna().any() or clean[column].lt(0).any():
            raise DataValidationError(f"{column} must contain non-negative values.")

    for column in (name for name in YES_NO_COLUMNS if name in model_inputs):
        values = clean[column].dropna().astype(str).str.strip().str.title()
        invalid_values = sorted(set(values) - {"Yes", "No"})
        if invalid_values:
            preview = ", ".join(invalid_values[:3])
            raise DataValidationError(f"{column} accepts only Yes or No; found: {preview}.")
        clean[column] = clean[column].astype("string").str.strip().str.title()

    for column, allowed_values in CATEGORICAL_VALUES.items():
        if column not in model_inputs:
            continue
        normalised = {value.casefold(): value for value in allowed_values}
        source = clean[column].astype("string").str.strip()
        canonical = source.str.casefold().map(normalised)
        invalid = source.notna() & canonical.isna()
        if invalid.any():
            examples = ", ".join(map(str, source[invalid].unique()[:3]))
            expected = ", ".join(allowed_values)
            raise DataValidationError(
                f"{column} contains unsupported values: {examples}. Expected: {expected}."
            )
        clean[column] = canonical.astype("string")

    return clean


def _clean_feature_names(columns: pd.Index) -> pd.Index:
    return (
        columns.astype(str)
        .str.replace(r"[\[\]<>]", "", regex=True)
        .str.replace(r"[(),]", "_", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )


def prepare_features(raw_df: pd.DataFrame, raw_feature_list: tuple[str, ...]) -> pd.DataFrame:
    """Apply the training-time transformation contract to validated rows."""
    dataframe = raw_df.drop(columns=list(LEAKAGE_COLUMNS), errors="ignore").copy()

    if TENURE_GROUP_COLUMN in raw_feature_list:
        dataframe[TENURE_GROUP_COLUMN] = pd.cut(
            dataframe[TENURE_COLUMN], bins=TENURE_BINS, include_lowest=True
        )

    features = dataframe[list(raw_feature_list)].copy()
    features = features.apply(
        lambda series: series.astype("object")
        if isinstance(series.dtype, pd.CategoricalDtype)
        else series
    )
    binary_map = {"Yes": 1, "No": 0}
    for column in features.columns:
        if pd.api.types.is_object_dtype(features[column]) or pd.api.types.is_string_dtype(
            features[column]
        ):
            values = set(features[column].dropna().unique())
            if values.issubset(binary_map):
                features[column] = features[column].map(binary_map)

    categorical_columns = features.select_dtypes(include=["object", "string"]).columns.tolist()
    if categorical_columns:
        # Encode every observed category, then align to the persisted schema. This keeps a
        # customer's encoding stable regardless of which other rows share the upload batch.
        features = pd.get_dummies(features, columns=categorical_columns, drop_first=False)

    if "Customer Satisfaction" in features.columns:
        features["Customer Satisfaction_missing"] = features[
            "Customer Satisfaction"
        ].isna().astype(int)
        features["Customer Satisfaction"] = features["Customer Satisfaction"].fillna(-1)

    features.columns = _clean_feature_names(features.columns)
    return features


def score_customers(
    raw_df: pd.DataFrame, artifacts: ModelArtifacts
) -> tuple[pd.DataFrame, list[str]]:
    """Return validated source rows with model scores and ignored-category warnings."""
    clean = validate_customer_data(raw_df, artifacts.raw_feature_list)
    prepared = prepare_features(clean, artifacts.raw_feature_list)
    ignored_columns = sorted(
        set(prepared.columns) - set(artifacts.training_columns) - BASELINE_DUMMY_COLUMNS
    )
    aligned = prepared.reindex(columns=artifacts.training_columns, fill_value=0)
    raw_scores = artifacts.model.predict_proba(aligned)[:, 1]
    scores = artifacts.calibrator.predict(raw_scores)

    if not np.isfinite(scores).all():
        raise RuntimeError("The model returned invalid probability values.")

    result = clean.copy()
    result["churn_risk_score"] = scores
    return result, ignored_columns
