# Story 3.1: Feature Audit & Selection (REVISED)

Status: done

## Story

As a Data Scientist,
I want to audit ALL features in the customer matrix and only discard truly useless ones,
so that the model preserves maximum behavioral information.

## Acceptance Criteria

1. Every feature from `customers_features.csv` (all 54 columns) classified as: **keep**, **drop**, or **transform**
2. Only drop: zero-variance features, exact duplicate columns, raw date columns, and traceability flags
3. Define 4 feature groups in `src/config.py`: `FEATURES_DROP`, `FEATURES_CONTINUOUS`, `FEATURES_ONEHOT`, `FEATURES_FREQUENCY`
4. At least one feature from each of the 6 families: RFM, Behavior, Product, Channel, Sociodemographic, Lifecycle
5. Markdown decision table with every feature and its fate

## Tasks / Subtasks

- [x] Task 1 — Define feature group constants in `src/config.py` (AC: 1–4)
  - [x] `FEATURES_DROP` — 14 features with justification per column
  - [x] `FEATURES_CONTINUOUS` — 30 numeric features → median impute + StandardScaler
  - [x] `FEATURES_ONEHOT` — 9 low-cardinality categoricals → `Unknown` + One-Hot
  - [x] `FEATURES_FREQUENCY` — 1 high-cardinality categorical → Frequency Encoding
  - [x] Legacy `CLUSTERING_FEATURES` retained for backward compatibility
- [x] Task 2 — Update notebook `02_clustering.ipynb` E3.1 section (AC: 5)
  - [x] Remplacer table E3.1 par audit complet 54 features
  - [x] Document 14 drops with rationale
  - [x] Show retained features by group (continuous, one-hot, frequency, indicators)

## Dev Notes

### Architecture Guardrails

**Module:** `src/config.py` — feature group constants replace old `CLUSTERING_FEATURES` for the preprocessing pipeline.
**Notebook:** `02_clustering.ipynb` — update E3.1 section.

**REVISED Decision Table (53 → 40 retained → 70 after encoding):**

| Category | Count | Features |
|---|---|---|
| **Dropped** | 14 | `is_new_customer` (zero var), `total_sales_eur`/`avg_sales_eur`/`total_transactions` (dupes), `salesVatEUR_first_purchase`/`first_purchase_channel` (dupes), `monetary_total_capped`/`frequency_capped` (winsorized dupes), `loyalty_status` (string of loyalty_numeric), `Axe_Desc_first_purchase` (raw), 3 dates, `is_outlier` |
| **Continuous** | 30 | RFM (4), Behavior (3), Volume (3), Channel ratios (3), Product ratios (5), Market ratios (4), Counts (4), Demographics (1), Lifecycle (2), Ordinal (1) |
| **One-Hot** | 9 | `gender`, `dominant_channel`, `dominant_axe`, `dominant_market`, `country`, `channel_recruitment`, `age_category`, `age_generation`, `first_purchase_axe` |
| **Frequency** | 1 | `customer_city` (~12K unique) |
| **Created** | 2 | `has_age_info`, `has_first_purchase_info` (missing indicators) |

**KEY CHANGE vs old story:** The old approach discarded `gender`, `age`, `country`, `customer_city`, `age_category`, `age_generation`, `channel_recruitment`, `dominant_*`, `first_purchase_axe`, `total_lines`, `total_discount_eur`, `total_quantity`, `nb_unique_brands`, `nb_unique_stores`, `market_sephora_ratio`, `market_others_ratio`, `first_purchase_amount`. The new approach keeps ALL of these with appropriate encoding.

### Previous Story Output (E2 complete)
- `data/processed/customers_features.csv` (54 columns, 64469 rows)

### Output of This Story
- `FEATURES_DROP`, `FEATURES_CONTINUOUS`, `FEATURES_ONEHOT`, `FEATURES_FREQUENCY` in `src/config.py`
- Updated notebook E3.1 section with comprehensive decision table

### References
- [epics — US-3.1 (REVISED)](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6 (GitHub Copilot)

### Debug Log References
No issues encountered.

### Completion Notes List
- **Code done:** `src/config.py` updated with 4 feature group constants + legacy `CLUSTERING_FEATURES` + missing `total_transactions` column logic
- **Tests fixed:** `tests/test_config.py` rewritten completely to test new feature groups, dropping legacy AC requirements. ACs 1-4 cover counts, duplication and family representation properly.
- **Notebook done:** E3.1 section updated — comprehensive 54-feature audit table, group breakdown code, all retained features validated
- Previously: 21 features selected, now: 40 retained (→ ~70 after encoding)

### File List
Files modified:
- `src/config.py` — added `FEATURES_DROP`, `FEATURES_CONTINUOUS`, `FEATURES_ONEHOT`, `FEATURES_FREQUENCY`
- `tests/test_config.py` — updated test suite for comprehensive groups
- `02_clustering.ipynb` — E3.1: updated import cell, comprehensive audit table (54 features), feature group summary code
