# Model Card: Telecom Churn Risk Model

## Summary

This XGBoost classifier ranks fictional telecom customers by predicted churn probability. It was
built to demonstrate an end-to-end analytics workflow and a human-reviewed retention process.

| Item | Detail |
|---|---|
| Model | XGBoost binary classifier |
| Artifact | XGBoost UBJSON (`churn_model.ubj`) |
| Training data | JB-Link fictional telco churn dataset |
| Target | `Churn Value` |
| Evaluation | Stratified 75/25 hold-out split, random state 42 |
| Primary metric | ROC-AUC and lift in the highest-risk deciles |
| Intended user | CRM, retention, customer operations, and analytics teams |

## Intended use

- Rank customers for human review when outreach capacity is limited.
- Explore how contract, service, price, tenure, and support signals relate to churn risk.
- Demonstrate how model output can feed a transparent operational decision layer.

The score must not autonomously determine customer eligibility, price, service access, or contact.

## Evaluation

The recorded hold-out ROC-AUC is `0.9755`. Churn in the highest-risk decile is `99.44%`, or `3.75x`
the portfolio rate. These point estimates come from a single random hold-out set and are likely
optimistic relative to a future, shifted customer population.

## Features and leakage controls

The model uses contract and tenure, charges, service configuration, usage, support interactions,
and customer satisfaction. `Churn Value`, `Churn Category`, and `Churn Reason` are explicitly
excluded from inference features. The application aligns every request to the persisted 40-column
training schema.

## Known limitations

- The dataset is fictional and is not representative evidence for New Zealand customers.
- Evaluation is a random hold-out rather than an out-of-time or external validation.
- The current probabilities are not calibrated for financial forecasting.
- Training data may encode historical sampling or service-design bias.
- Recommended actions are rules, not causal treatment-effect estimates.
- No protected-attribute fairness claims are made.

## Production requirements

Before live use, an owner would need to establish representative data, consent and retention rules,
time-based evaluation, probability calibration, segment-level performance and fairness checks,
drift and data-quality monitoring, an intervention hold-out, access controls, audit logging, model
versioning, rollback, and a documented human escalation process.

## Monitoring suggestions

- Input schema, missingness, categorical novelty, and score distribution
- ROC-AUC, precision/recall, lift, and calibration after labels mature
- Performance by contract, tenure, geography, and reviewed protected groups
- Contact rate, offer rate, save rate, incremental retention, and cost per save
- Override rate, complaint rate, and unintended customer outcomes

## Version

This card describes the `v1.0.0` portfolio baseline.
