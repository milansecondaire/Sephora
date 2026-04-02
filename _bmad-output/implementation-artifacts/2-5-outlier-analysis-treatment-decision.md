# Story 2.5: Outlier Analysis & Treatment Decision

Status: done

## Story

As a Data Analyst,
I want to identify extreme customers (very high spenders, very high frequency),
so that I can decide whether to include, cap, or exclude them from clustering.

## Acceptance Criteria

1. Box plots for `monetary_total`, `frequency`, `recency_days`
2. Customers beyond 99th percentile on `monetary_total` listed with their key stats
3. Decision documented: cap at 99th percentile (Winsorization) OR keep OR separate analysis
4. Chosen strategy applied and a column `is_outlier` created for traceability

## Tasks / Subtasks

- [x] Task 1 — Implement `plot_outlier_boxplots()` in `src/visualization.py` (AC: 1)
  - [x] Box plots for `monetary_total`, `frequency`, `recency_days` side by side
  - [x] Save to `figures/outlier_boxplots.png`
- [x] Task 2 — Identify extreme customers (AC: 2)
  - [x] Compute 99th percentile for `monetary_total`
  - [x] Filter customers above threshold; display as table with stats
- [x] Task 3 — Apply outlier strategy and create `is_outlier` column (AC: 3, 4)
  - [x] Apply Winsorization (cap at 99th percentile) to `monetary_total`, `frequency`
  - [x] Create `is_outlier = True` for customers initially above the cap
  - [x] Update `customers_features.csv` with `is_outlier` column
- [x] Task 4 — Add notebook section in `first-analysis.ipynb` with decision Markdown

## Dev Notes

### Architecture Guardrails

**Module:** `src/visualization.py` + inline notebook logic (outlier strategy is an analytical decision, not reusable module logic).  
**Notebook:** `01_eda.ipynb` — E2 section, after US-2.4.

**Architecture decision D2.4:**
> "Retain all customers in the primary clustering.
> DBSCAN used specifically for noise/outlier detection.
> Exclusion only if DBSCAN noise AND < 0.5% of base."

The **recommended default strategy** is Winsorization (cap values) rather than exclusion. Document clearly in Markdown.

**Winsorization pattern:**
```python
from scipy.stats.mstats import winsorize
p99_monetary = df_customers['monetary_total'].quantile(0.99)
p99_frequency = df_customers['frequency'].quantile(0.99)
df_customers['is_outlier'] = (
    (df_customers['monetary_total'] > p99_monetary) |
    (df_customers['frequency'] > p99_frequency)
)
# Cap (winsorize) — do NOT drop
df_customers['monetary_total_capped'] = df_customers['monetary_total'].clip(upper=p99_monetary)
df_customers['frequency_capped'] = df_customers['frequency'].clip(upper=p99_frequency)
```

**Re-save `customers_features.csv`** after adding `is_outlier`:
```python
df_customers.to_csv(DATA_PROCESSED_PATH + "customers_features.csv")
```

### Previous Story Output (2-4)
- `df_customers` with all features

### Output of This Story
- `figures/outlier_boxplots.png`
- `df_customers` with `is_outlier`, `monetary_total_capped`, `frequency_capped` columns
- Updated `data/processed/customers_features.csv`

### References
- [epics — US-2.5](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — D2.4 Outlier Handling Strategy](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6 (Amelia Dev Agent)

### Debug Log References
None — all tasks completed without errors.

### Completion Notes List
- `plot_outlier_boxplots()`: side-by-side box plots for monetary_total, frequency, recency_days
- `identify_extreme_customers()`: returns customers above p99 threshold + threshold value
- `apply_winsorization()`: caps monetary_total & frequency at p99, creates `is_outlier` flag + `*_capped` columns
- 99th percentile monetary_total = 1,452.62 EUR; 1.38% customers flagged as outliers (887/64,469)
- Strategy: Winsorization (cap, not remove) — aligned with architecture D2.4
- 18 tests added and passing for US 2-5 functions

### File List
Files modified:
- `src/visualization.py` (added `plot_outlier_boxplots()`, `identify_extreme_customers()`, `apply_winsorization()`)
- `first-analysis.ipynb` (added US-2.5 section: cells E2-5a, E2-5b, E2-5c + decision markdown)
- `tests/test_visualization.py` (added TestPlotOutlierBoxplots, TestIdentifyExtremeCustomers, TestApplyWinsorization)
- `data/processed/customers_features.csv` (updated with is_outlier, monetary_total_capped, frequency_capped)
