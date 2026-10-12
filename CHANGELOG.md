# Changelog

All notable changes to this project are documented here.

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
