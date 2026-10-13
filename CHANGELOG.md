# Changelog

## [2.0.0] - 2026-10-13

### Changed

- Reframed the repository around ML methodology and evidence rather than the downstream UI.
- Replaced test-set tuning with a stratified 60/20/20 train/validation/test contract.
- Added five-fold comparison of logistic regression, random forest, and two XGBoost candidates.
- Excluded Customer Satisfaction because of ambiguous timing and 74% missingness.
- Added validation-only Platt calibration and cost-aware threshold selection.
- Added untouched-test ranking, calibration, threshold, uncertainty, and SHAP evidence.
- Updated inference to serve calibrated probabilities and added an ML evidence view to the app.

### Added

- Reproducible `scripts/train_model.py` entry point and CI model rebuild.
- Regression tests for calibration, published evaluation evidence, and split integrity.

All notable changes to this project are documented here.

## [1.0.1] - 2026-10-13

### Fixed

- Replaced deprecated Streamlit width arguments for warning-free operation on 1.65.0

## [1.0.0] - 2026-10-13

### Added

- Built-in 120-customer demo portfolio
- Capacity-aware retention action queue and CSV export
- Risk tiers, revenue-at-risk estimates, and transparent action recommendations
- Guarded CSV validation with actionable user messages
- Desktop and mobile responsive presentation
- Unit, regression, batch-stability, and Streamlit smoke tests
- Ruff checks and split CI jobs for application quality and notebook reproducibility
- Model card, product screenshots, architecture, and local development documentation

### Changed

- Extracted preprocessing, scoring, and decision logic into a reusable `src/` package
- Replaced batch-dependent category encoding with a fixed training feature contract
- Migrated the XGBoost model from a Python pickle to portable UBJSON

### Security

- Added upload size and row limits
- Rejected malformed schemas, duplicate identifiers, invalid numeric data, and unsupported categories
- Replaced raw exception output with user-safe error messages
- Upgraded Streamlit and nbconvert to patched releases covering all known repository alerts
