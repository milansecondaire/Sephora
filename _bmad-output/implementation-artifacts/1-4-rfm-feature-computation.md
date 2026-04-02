# Story 1.4: RFM Feature Computation

Status: done

## Story

As a Data Analyst,
I want to compute Recency, Frequency, and Monetary features per customer,
so that the backbone of behavioral segmentation is established.

## Acceptance Criteria

1. Reference date set to `2025-12-31`
2. `recency_days` = (2025-12-31 − `last_purchase_date`).days; integer, ≥ 0
3. `frequency` = `total_transactions` (number of unique tickets)
4. `monetary_total` = `total_sales_eur` (sum of all purchases)
5. `monetary_avg` = `avg_sales_eur` (mean basket in EUR)
6. All four features have no null values
7. Summary statistics printed: `df_customers[['recency_days','frequency','monetary_total','monetary_avg']].describe()`

## Tasks / Subtasks

- [x] Task 1 — Implement `compute_rfm_features()` in `src/feature_engineer.py` (AC: 1–6)
  - [x] Import `RECENCY_REFERENCE_DATE` from `src.config`
  - [x] Compute `recency_days` from reference date and `last_purchase_date`
  - [x] Cast `recency_days` to `int`
  - [x] Assign `frequency` = `total_transactions`
  - [x] Assign `monetary_total` = `total_sales_eur`
  - [x] Assign `monetary_avg` = `avg_sales_eur`
  - [x] Assert zero nulls on all four columns
  - [x] Return modified `df_customers`
- [x] Task 2 — Add notebook section in `first-analysis.ipynb` (AC: 7)
  - [x] Call `compute_rfm_features(df_customers)` (in-place or reassign)
  - [x] Display `.describe()` output
  - [x] Markdown cell: what each RFM column represents

## Dev Notes

### Architecture Guardrails

**Module:** `src/feature_engineer.py` — add to existing module.  
**Notebook:** `01_eda.ipynb` — E1 section, after US-1.3 cells.  
**Column naming convention:** `rfm_` prefix is used in the FINAL scaled matrix (E3), but here at E1 level the raw column names `recency_days`, `frequency`, `monetary_total`, `monetary_avg` are used directly on `df_customers`.

**Reference date (from config):**
```python
from src.config import RECENCY_REFERENCE_DATE
import pandas as pd

ref_date = pd.Timestamp(RECENCY_REFERENCE_DATE)  # 2025-12-31
df_customers['recency_days'] = (ref_date - df_customers['last_purchase_date']).dt.days.astype(int)
```

**`RECENCY_REFERENCE_DATE` is already defined in `src/config.py`** (created in US-1.1) as `"2025-12-31"`.

**Zero-null assertion:**
```python
rfm_cols = ['recency_days', 'frequency', 'monetary_total', 'monetary_avg']
assert df_customers[rfm_cols].isnull().sum().sum() == 0, "RFM features contain nulls"
```

### Previous Story Output (1-3)
- `df_customers`: customer-level DataFrame, index=`anonymized_card_code`
- Columns available: `total_transactions`, `last_purchase_date`, `total_sales_eur`, `avg_sales_eur`

### Output of This Story
- `df_customers` with added columns: `recency_days`, `frequency`, `monetary_total`, `monetary_avg`

### References
- [epics — US-1.4](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — D2.1 Centralized Configuration (RECENCY_REFERENCE_DATE)](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
No issues encountered.

### Completion Notes List
- Implemented `compute_rfm_features()` in `src/feature_engineer.py`: computes `recency_days`, `frequency`, `monetary_total`, `monetary_avg` with zero-null assertion
- Added 11 unit tests in `TestComputeRfmFeatures` class — all pass (50/50 total suite)
- Added Code cell + Markdown cell in `first-analysis.ipynb` (notebook is named `first-analysis.ipynb`, not `01_eda.ipynb`)
- Red-green-refactor cycle followed: 11 tests RED → implementation → 50/50 GREEN

**Review Follow-up Notes (AI):**
- ✅ Implemented `.copy()` in `compute_rfm_features()` to avoid setting values on a slice and unintended object mutation.
- ✅ Added constraint logic `np.maximum(0, raw_recency)` to enforce non-negative recency correctly according to AC 2.
- ✅ Moved null-check `assert df_rfm["last_purchase_date"].isnull().sum() == 0` early in the function to avoid `IntCastingNaNError` exceptions.
- ✅ Added 3 new edge-case tests validating constraints: `test_does_not_mutate_original`, `test_recency_days_clipped_to_zero`, and `test_raises_error_on_missing_last_purchase_date`. All 53 tests passing.

### File List
- `src/feature_engineer.py` — added `compute_rfm_features()`, imported `RECENCY_REFERENCE_DATE`
- `tests/test_feature_engineer.py` — added `customer_df` fixture + `TestComputeRfmFeatures` (11 tests)
- `first-analysis.ipynb` — added Cell 6 (code) + RFM explanation markdown cell
