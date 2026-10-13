# Model Card: Telecom Churn Risk Model v2.0.0

## Model details

| Item | Detail |
|---|---|
| Task | Binary churn classification and risk ranking |
| Estimator | Shallow XGBoost selected from four predeclared candidates |
| Probability layer | Platt calibration fitted on validation only |
| Training data | 7,043-row JB-Link fictional telecom dataset |
| Target | `Churn Value` |
| Split | Stratified 60% train / 20% validation / 20% test, seed 42 |
| Selection | Five-fold CV on train; primary metric ROC-AUC |
| Artifact | XGBoost UBJSON plus persisted calibrator and feature schema |

## Intended use

The model demonstrates how pre-decision CRM and service signals can rank customers for human
review when retention capacity is limited. It may be used for portfolio learning, interviews, and
technical evaluation. It must not autonomously determine eligibility, price, service, or contact.

## Evaluation

The untouched 1,409-row test set produced ROC-AUC `0.9574` (95% bootstrap CI
`0.9452-0.9682`) and PR-AUC `0.9181` (`0.8958-0.9382`). Calibrated Brier score is `0.0645`.

A demonstration policy prices a false negative at five false positives. Its validation-selected
threshold is `0.1049`; test recall is `94.1%`, precision `61.2%`, specificity `78.5%`, and F2
`0.8498`. This operating point must be replaced by organisation-specific capacity, intervention
cost, and benefit estimates before any real use.

## Data and leakage controls

Direct outcome fields (`Churn Category` and `Churn Reason`) are excluded. `Customer Satisfaction`
is also excluded because 74% of values are missing and its timing relative to churn is ambiguous.
Customer identifiers, demographics, location, and CLTV are not model inputs. The saved split
manifest makes customer-level partition membership auditable.

Service requests and issue counts are assumed to exist before the score timestamp. A production
feature pipeline would need point-in-time correct joins to enforce that assumption.

## Interpretation

Native XGBoost SHAP contributions are aggregated on test data. They describe which encoded inputs
most influence model output, not the direction or causal effect of an intervention. Recommended
actions in the demo are deterministic rules and are not model explanations.

## Known limitations

- Fictional data is not evidence of New Zealand population performance.
- Random splits may overstate performance relative to future cohorts.
- No external or out-of-time dataset is available.
- Calibration may drift as customer mix, products, and pricing change.
- The dataset cannot support a defensible protected-group fairness assessment.
- Retention effectiveness requires treatment-effect measurement, not churn prediction alone.

## Monitoring requirements for real use

- Point-in-time feature availability, schema, missingness, and unseen categories
- Population stability and score-distribution drift
- ROC-AUC, PR-AUC, Brier score, calibration, recall, and precision after labels mature
- Segment performance and fairness with appropriately governed attributes
- Contact rate, incremental save rate, cost per save, complaints, and overrides
- Versioned data, code, model, calibrator, threshold, approvals, and rollback evidence

## Reproducibility

`python scripts/train_model.py` rebuilds all model and evaluation artifacts. Dataset SHA-256,
candidate results, split sizes, test metrics, threshold policy, and SHAP rankings are persisted in
`artifacts/model_metadata.json`.
