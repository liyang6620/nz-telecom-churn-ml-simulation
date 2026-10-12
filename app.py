"""Streamlit interface for customer churn decision support."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from churn_app.config import DEMO_DATA_PATH, MAX_UPLOAD_BYTES  # noqa: E402
from churn_app.decisions import RISK_ORDER, build_retention_plan  # noqa: E402
from churn_app.scoring import DataValidationError, load_artifacts, score_customers  # noqa: E402

st.set_page_config(page_title="Retention control centre", page_icon="◫", layout="wide")

st.markdown(
    """
    <style>
    :root { --ink: #17211b; --muted: #627069; --line: #dce4df; --green: #16754a; }
    .stApp { background: #f7f9f7; color: var(--ink); }
    .block-container { max-width: 1240px; padding-top: 4rem; padding-bottom: 3rem; }
    h1, h2, h3, p, label, [data-testid="stMetricLabel"] { letter-spacing: 0 !important; }
    h1 { font-size: 2rem !important; line-height: 1.18 !important; }
    h2 { font-size: 1.3rem !important; margin-top: 0.6rem !important; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid var(--line);
                               padding: 1rem; border-radius: 6px; }
    [data-testid="stMetricValue"] { font-size: 1.55rem; }
    [data-testid="stSidebar"] { border-right: 1px solid var(--line); }
    .eyebrow { color: var(--green); font-size: .78rem; font-weight: 700;
               text-transform: uppercase; }
    .subhead { color: var(--muted); font-size: 1rem; margin: -.35rem 0 1.25rem; }
    .status { display: inline-block; border: 1px solid #b9d8c7; color: #11613d; background: #edf8f1;
              border-radius: 999px; padding: .22rem .65rem; font-size: .78rem; font-weight: 650; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 6px;
                                  overflow: hidden; }
    .stButton button, .stDownloadButton button { min-height: 2.65rem; border-radius: 5px; }
    @media (max-width: 700px) {
      .block-container { padding: 4rem .85rem 2rem; }
      h1 { font-size: 1.7rem !important; }
      [data-testid="stMetric"] { padding: .8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_data
def load_demo_data() -> pd.DataFrame:
    return pd.read_csv(DEMO_DATA_PATH)


def read_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        raise DataValidationError("The upload exceeds the 10 MB demo limit.")
    try:
        return pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise DataValidationError("The file is not a readable UTF-8 CSV.") from error


try:
    artifacts = get_artifacts()
except Exception:
    st.error("The model bundle is unavailable. Please try again later.")
    st.stop()

with st.sidebar:
    st.markdown("### Scoring setup")
    source_mode = st.radio("Customer source", ["Built-in demo", "Upload CSV"])
    uploaded = None
    if source_mode == "Upload CSV":
        uploaded = st.file_uploader("Customer snapshot", type=["csv"])
        st.caption("UTF-8 CSV · up to 10 MB · up to 50,000 rows")
        with open(DEMO_DATA_PATH, "rb") as template_file:
            st.download_button(
                "Download input template",
                template_file,
                file_name="customer_snapshot_template.csv",
                mime="text/csv",
                width="stretch",
            )

try:
    if source_mode == "Built-in demo":
        raw_data = load_demo_data()
    elif uploaded is None:
        st.info("Upload a customer snapshot to begin, or switch to the built-in demo.")
        st.stop()
    else:
        raw_data = read_upload(uploaded)

    scored, ignored_categories = score_customers(raw_data, artifacts)
except DataValidationError as error:
    st.error(str(error))
    st.caption(
        "Use the downloadable template to check the required column names and value formats."
    )
    st.stop()
except Exception:
    st.error("Scoring could not be completed. Check the file and try again.")
    st.stop()

customer_count = len(scored)
default_capacity = min(max(1, round(customer_count * 0.2)), customer_count)
with st.sidebar:
    st.divider()
    contact_capacity = st.number_input(
        "Contacts this cycle",
        min_value=1,
        max_value=customer_count,
        value=default_capacity,
        help="Sets the number of highest-risk customers included in the action queue.",
    )
    st.caption("Scores support human decisions; they do not trigger automated customer action.")

plan = build_retention_plan(scored, int(contact_capacity))
queue = plan.loc[plan["contact_this_cycle"]].copy()

st.markdown('<div class="eyebrow">NZ telecom churn simulation</div>', unsafe_allow_html=True)
st.title("Retention control centre")
st.markdown(
    '<div class="subhead">Turn churn probabilities into a capacity-aware customer '
    "action queue.</div>",
    unsafe_allow_html=True,
)
source_label = "Demo data" if source_mode == "Built-in demo" else "Uploaded data"
st.markdown(
    f'<span class="status">Model ready · {source_label}</span>', unsafe_allow_html=True
)

critical_count = int(plan["risk_tier"].eq("Critical").sum())
average_risk = plan["churn_risk_score"].mean()
revenue_at_risk = queue["monthly_revenue_at_risk"].sum()

metric_columns = st.columns(4)
metric_columns[0].metric("Customers scored", f"{customer_count:,}")
metric_columns[1].metric("Critical risk", f"{critical_count:,}")
metric_columns[2].metric("Average churn risk", f"{average_risk:.1%}")
metric_columns[3].metric("Queue revenue at risk", f"${revenue_at_risk:,.0f}/mo")

overview_tab, queue_tab, all_results_tab, notes_tab = st.tabs(
    ["Overview", "Action queue", "All results", "Notes"]
)

with overview_tab:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Customers by risk tier")
        tier_counts = (
            plan["risk_tier"].value_counts().reindex(RISK_ORDER, fill_value=0).rename("Customers")
        )
        st.bar_chart(tier_counts, color="#16754a", horizontal=True, height=300)
    with right:
        st.subheader("Average risk by contract")
        contract_risk = (
            plan.groupby("Contract", dropna=False)["churn_risk_score"]
            .mean()
            .sort_values(ascending=True)
            .rename("Average risk")
        )
        st.bar_chart(contract_risk, color="#c05a32", horizontal=True, height=300)
    st.caption(
        "Risk tiers: Critical ≥ 75%, High 50–74%, Medium 25–49%, Low < 25%. "
        "Revenue at risk is monthly charge × predicted churn probability."
    )

with queue_tab:
    st.subheader(f"This cycle's {len(queue):,} priority contacts")
    st.caption(
        "Ranked by predicted churn probability; recommended actions use transparent business rules."
    )
    queue_columns = [
        "priority_rank",
        "Customer ID",
        "risk_tier",
        "churn_risk_score",
        "Monthly Charge",
        "monthly_revenue_at_risk",
        "Contract",
        "recommended_action",
    ]
    st.dataframe(
        queue[queue_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "priority_rank": st.column_config.NumberColumn("Rank", format="%d"),
            "risk_tier": "Risk tier",
            "churn_risk_score": st.column_config.ProgressColumn(
                "Churn risk", min_value=0, max_value=1, format="percent"
            ),
            "Monthly Charge": st.column_config.NumberColumn("Monthly charge", format="$%.2f"),
            "monthly_revenue_at_risk": st.column_config.NumberColumn(
                "Revenue at risk", format="$%.2f"
            ),
            "recommended_action": "Recommended action",
        },
    )

with all_results_tab:
    st.subheader("Complete scored portfolio")
    display_columns = [
        "priority_rank",
        "Customer ID",
        "risk_tier",
        "churn_risk_score",
        "contact_this_cycle",
        "monthly_revenue_at_risk",
        "recommended_action",
    ]
    st.dataframe(
        plan[display_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "priority_rank": "Rank",
            "risk_tier": "Risk tier",
            "churn_risk_score": st.column_config.ProgressColumn(
                "Churn risk", min_value=0, max_value=1, format="percent"
            ),
            "contact_this_cycle": "In queue",
            "monthly_revenue_at_risk": st.column_config.NumberColumn(
                "Revenue at risk", format="$%.2f"
            ),
            "recommended_action": "Recommended action",
        },
    )
    st.download_button(
        "Download scored portfolio",
        plan.to_csv(index=False).encode("utf-8"),
        file_name="churn_retention_plan.csv",
        mime="text/csv",
        type="primary",
    )

with notes_tab:
    st.subheader("How to interpret this demo")
    st.markdown(
        """
        - **Churn risk** is an XGBoost probability used to rank customers, not a guarantee of churn.
        - **Risk tier** and **recommended action** are transparent rules layered on the score.
        - **Revenue at risk** is a prioritisation estimate, not a financial forecast.
        - The fictional public dataset has not been validated on real NZ customers.
        - Human review is required before any offer or customer contact.
        """
    )
    if ignored_categories:
        st.warning(
            "Some category values were not present in the training schema and were treated as "
            "unseen: " + ", ".join(ignored_categories[:8])
        )
