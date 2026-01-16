# app.py
# Lightweight Streamlit frontend for churn risk scoring
# - Loads persisted model artifacts
# - Applies the same preprocessing rules used during training (Section 7.6)
# - Outputs churn risk scores for uploaded customer snapshots

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Artifact filenames

MODEL_PATH = "churn_model.pkl"
SCHEMA_PATH = "training_schema.pkl"
RAW_FEATURES_PATH = "raw_feature_list.pkl"

# Leakage columns to drop if present
LEAKAGE_COLS = ["Churn Value", "Churn Category", "Churn Reason"]

# Keep consistent with training
TENURE_BINS = [0, 6, 12, 24, 36, 48, 60, 72]
TENURE_COL = "Tenure in Months"
TENURE_GROUP_COL = "tenure_group"

def _clean_feature_names(cols: pd.Index) -> pd.Index:
    return (
        cols.astype(str)
        .str.replace(r"[\[\]<>]", "", regex=True)
        .str.replace(r"[(),]", "_", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )

def _prepare_features_for_scoring(raw_df: pd.DataFrame, raw_feature_list: list[str]) -> pd.DataFrame:
    df = raw_df.copy()

    # 0) Drop leakage columns
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns], errors="ignore")

    # 1) Derive tenure_group if used
    if TENURE_GROUP_COL in raw_feature_list:
        if TENURE_COL not in df.columns:
            raise ValueError(f"Missing required column '{TENURE_COL}'.")
        df[TENURE_GROUP_COL] = pd.cut(
            df[TENURE_COL],
            bins=TENURE_BINS,
            include_lowest=True
        )

    # 2) Keep only training raw features
    missing = [c for c in raw_feature_list if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    X_new = df[raw_feature_list].copy()
        # 3) Yes/No mapping
    binary_map = {"Yes": 1, "No": 0}
    for col in X_new.columns:
        if X_new[col].dtype == "object":
            vals = set(X_new[col].dropna().unique())
            if vals.issubset({"Yes", "No"}):
                X_new[col] = X_new[col].map(binary_map)

    # 4) One-hot encode
    cat_cols = X_new.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        X_new = pd.get_dummies(X_new, columns=cat_cols, drop_first=True)

    # 5) Customer Satisfaction missingness
    if "Customer Satisfaction" in X_new.columns:
        X_new["Customer Satisfaction_missing"] = X_new["Customer Satisfaction"].isna().astype(int)
        X_new["Customer Satisfaction"] = X_new["Customer Satisfaction"].fillna(-1)

    X_new.columns = _clean_feature_names(X_new.columns)
    return X_new

def score_customers(raw_df, model, training_columns, raw_feature_list):
    X_new = _prepare_features_for_scoring(raw_df, raw_feature_list)
    X_new = X_new.reindex(columns=training_columns, fill_value=0)
    scores = model.predict_proba(X_new)[:, 1]
    return pd.Series(scores, index=raw_df.index, name="churn_risk_score")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    training_columns = pd.Index(joblib.load(SCHEMA_PATH))
    raw_feature_list = joblib.load(RAW_FEATURES_PATH)
    return model, training_columns, raw_feature_list

try:
    model, training_columns, raw_feature_list = load_artifacts()
except Exception as e:
    st.error("Failed to load model artifacts.")
    st.exception(e)
    st.stop()

st.set_page_config(page_title="Churn Risk Scoring", layout="wide")
st.title("Churn Risk Scoring")
st.caption("Upload customer snapshot → score churn risk → export results")

uploaded = st.file_uploader("Upload customer snapshot CSV", type=["csv"])
if uploaded is None:
    st.stop()

raw_df = pd.read_csv(uploaded)

scores = score_customers(raw_df, model, training_columns, raw_feature_list)

result_df = raw_df.copy()
result_df["churn_risk_score"] = scores
result_df = result_df.sort_values("churn_risk_score", ascending=False)

st.dataframe(result_df.head(50))
st.download_button(
    "Download scored CSV",
    result_df.to_csv(index=False),
    "scored_customers.csv"
)

