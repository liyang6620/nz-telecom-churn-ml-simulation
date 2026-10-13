"""Shared model and application configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "churn_model.ubj"
CALIBRATOR_PATH = PROJECT_ROOT / "probability_calibrator.pkl"
SCHEMA_PATH = PROJECT_ROOT / "training_schema.pkl"
RAW_FEATURES_PATH = PROJECT_ROOT / "raw_feature_list.pkl"
MODEL_METADATA_PATH = PROJECT_ROOT / "artifacts" / "model_metadata.json"
DEMO_DATA_PATH = PROJECT_ROOT / "data" / "telco_test_sample.csv"

CUSTOMER_ID_COLUMN = "Customer ID"
LEAKAGE_COLUMNS = ("Churn Value", "Churn Category", "Churn Reason")
TENURE_COLUMN = "Tenure in Months"
TENURE_GROUP_COLUMN = "tenure_group"
TENURE_BINS = (0, 6, 12, 24, 36, 48, 60, 72)

# Only features available at the scoring point belong here. Customer Satisfaction is deliberately
# excluded: it is missing for 74% of rows and its collection timing relative to churn is unclear.
MODEL_FEATURES = (
    "Contract",
    "Tenure in Months",
    "Offer",
    TENURE_GROUP_COLUMN,
    "Monthly Charge",
    "Total Regular Charges",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "Avg Monthly Long Distance Charges",
    "Paperless Billing",
    "Payment Method",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Internet Type",
    "Unlimited Data",
    "Avg Monthly GB Download",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Total Customer Svc Requests",
    "Product/Service Issues Reported",
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ROWS = 50_000

NUMERIC_COLUMNS = (
    "Tenure in Months",
    "Monthly Charge",
    "Total Regular Charges",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "Avg Monthly Long Distance Charges",
    "Avg Monthly GB Download",
    "Total Customer Svc Requests",
    "Product/Service Issues Reported",
    "Customer Satisfaction",
)

YES_NO_COLUMNS = (
    "Paperless Billing",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Unlimited Data",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
)

CATEGORICAL_VALUES = {
    "Contract": ("Month-to-Month", "One Year", "Two Year"),
    "Offer": ("Offer A", "Offer B", "Offer C", "Offer D", "Offer E"),
    "Payment Method": ("Bank Withdrawal", "Credit Card", "Mailed Check"),
    "Internet Type": ("Cable", "DSL", "Fiber Optic"),
}

BASELINE_DUMMY_COLUMNS = {
    "Contract_Month-to-Month",
    "Offer_Offer_A",
    "Payment_Method_Bank_Withdrawal",
    "Internet_Type_Cable",
    "tenure_group_-0.001_6.0",
}
