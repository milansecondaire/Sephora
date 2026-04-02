# Story 1.7: Lifecycle Feature Computation

Status: complete

## Story

As a Data Analyst,
I want to compute customer lifecycle and loyalty features,
so that the clustering can distinguish new, active, and mature customers.

## Acceptance Criteria

1. `subscription_tenure_days` = (2025-12-31 − `subscription_date`).days; NaN where `subscription_date` is missing
2. `loyalty_numeric` = ordinal encoding: No Fid=0, BRONZE=1, SILVER=2, GOLD=3
3. `is_new_customer` = 1 if `first_purchase_date` ≥ 2025-01-01, else 0
4. `first_purchase_axe` = dominant axis from `Axe_Desc_first_purchase` (parse list string, take first non-null element)
5. `first_purchase_channel` = `channel_recruitment` (already scalar)
6. `first_purchase_amount` = `salesVatEUR_first_purchase`
7. `customers_features.csv` saved to `data/processed/customers_features.csv` — this is the E1 final output
8. Shape of `customers_features.csv` printed: `(n_customers, n_features)`

## Tasks / Subtasks

- [x] Task 1 — Implement `compute_lifecycle_features()` in `src/feature_engineer.py` (AC: 1–6)
  - [x] Compute `subscription_tenure_days` with NaN where subscription_date is null
  - [x] Build `loyalty_numeric` from `loyalty_status` map
  - [x] Set `is_new_customer` based on `first_purchase_date`
  - [x] Parse `first_purchase_axe` from `Axe_Desc_first_purchase` (handle list-like string or scalar)
  - [x] Copy `channel_recruitment` → `first_purchase_channel`
  - [x] Copy `salesVatEUR_first_purchase` → `first_purchase_amount`
  - [x] Return `df_customers` with lifecycle columns added
- [x] Task 2 — Save `customers_features.csv` (AC: 7, 8)
  - [x] `save_customers_features()` function created; calls `to_csv()` and prints shape
  - [x] Print shape; confirm file saved
- [x] Task 3 — Add notebook section in `first-analysis.ipynb`
  - [x] Call `compute_lifecycle_features(df_customers)`
  - [x] Save the CSV
  - [x] Markdown cell: summary of E1 pipeline — total features now in `df_customers`

## Dev Notes

### Architecture Guardrails

**Module:** `src/feature_engineer.py` — final E1 function, add to existing module.  
**Notebook:** `01_eda.ipynb` — E1 final section. After this story, E1 is complete.

**Output file path (from config):**
```python
from src.config import DATA_PROCESSED_PATH, RECENCY_REFERENCE_DATE
import pandas as pd, os
os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
output_path = DATA_PROCESSED_PATH + "customers_features.csv"
df_customers.to_csv(output_path)
print(f"Saved: {output_path} — shape: {df_customers.shape}")
```

**subscription_tenure_days:**
```python
ref_date = pd.Timestamp(RECENCY_REFERENCE_DATE)
df_customers['subscription_tenure_days'] = (
    ref_date - df_customers['subscription_date']
).dt.days  # NaN preserved where subscription_date is NaT
```

**loyalty_numeric mapping:**
```python
LOYALTY_MAP = {'No Fid': 0, 'BRONZE': 1, 'SILVER': 2, 'GOLD': 3}
df_customers['loyalty_numeric'] = df_customers['loyalty_status'].map(LOYALTY_MAP)
```

**is_new_customer:**
```python
df_customers['is_new_customer'] = (
    df_customers['first_purchase_date'] >= pd.Timestamp('2025-01-01')
).astype(int)
```

**first_purchase_axe** — `Axe_Desc_first_purchase` may contain a list-like string (e.g., `"['MAKE UP']"`) or a plain string. Parse robustly:
```python
import ast
def parse_first_axe(val):
    if pd.isna(val):
        return np.nan
    try:
        parsed = ast.literal_eval(str(val))
        return parsed[0] if isinstance(parsed, list) and len(parsed) > 0 else str(val)
    except Exception:
        return str(val)

df_customers['first_purchase_axe'] = df_customers['Axe_Desc_first_purchase'].apply(parse_first_axe)
```

**E1 pipeline summary:** After this story, `df_customers` contains ~40+ feature columns across families: RFM, behavioral, channel, product affinity, sociodemographic, lifecycle. This is the master feature matrix consumed by E2 (EDA) and E3 (preprocessing).

### Previous Story Output (1-6)
- `df_customers` with product affinity + behavioral + channel + RFM + aggregation columns
- All prior feature computation complete

### Output of This Story (E1 Final Output)
- `df_customers` with lifecycle columns added
- `data/processed/customers_features.csv` — the single source of truth for E2 and E3

### References
- [epics — US-1.7](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — D1.1 Intermediate Storage (customers_features.csv)](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — D2.1 Centralized Configuration (DATA_PROCESSED_PATH)](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
None

### Completion Notes List
- `compute_lifecycle_features()` + `save_customers_features()` + `_parse_first_axe()` helper added
- `LOYALTY_MAP` constant added at module level
- 14 unit tests (12 lifecycle + 2 save) in `tests/test_feature_engineer_stories_5_6_7.py` — all pass
- Notebook cells added: Cell 9 (lifecycle), Cell 10 (save CSV), markdown summary

### File List
Files modified:
- `src/feature_engineer.py` (added `compute_lifecycle_features()`, `save_customers_features()`, `_parse_first_axe()`)
- `first-analysis.ipynb` (added US-1.7 section + CSV save + E1 summary)
- `tests/test_feature_engineer_stories_5_6_7.py` (14 tests for US-1.7)

Files created by this story:
- `data/processed/customers_features.csv` (created at notebook runtime)
