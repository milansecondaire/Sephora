# Story 1.5: Behavioral & Channel Feature Computation

Status: complete

## Story

As a Data Analyst,
I want to compute purchase behavior and channel preference features,
so that the clustering captures shopping patterns beyond pure RFM.

## Acceptance Criteria

1. `avg_basket_size_eur` = `total_sales_eur` / `total_transactions`
2. `avg_units_per_basket` = `total_quantity` / `total_transactions`
3. `discount_rate` = `total_discount_eur` / `total_sales_eur` (capped at 1.0; 0 where no discount)
4. `store_ratio` = share of transactions with `channel == 'store'` (excluding C&C)
5. `estore_ratio` = share of `channel == 'estore'` (excluding C&C)
6. `click_collect_ratio` = share of C&C transactions
7. `dominant_channel` = argmax of (store_ratio, estore_ratio, click_collect_ratio)
8. `nb_unique_brands` = number of distinct brands purchased
9. `nb_unique_stores` = number of distinct `store_code_name` visited

## Tasks / Subtasks

- [x] Task 1 — Compute basket and discount ratios per customer from `df_clean` (AC: 1–3)
  - [x] `avg_basket_size_eur` = `total_sales_eur` / `total_transactions`
  - [x] `avg_units_per_basket` = `total_quantity` / `total_transactions`
  - [x] `discount_rate`: handle zero-sales case; cap at 1.0 with `.clip(upper=1.0)`
- [x] Task 2 — Compute channel ratios from transaction data (AC: 4–7)
  - [x] Join back to `df_clean` to count per-customer channel distribution
  - [x] `store_ratio`: transactions where `channel == 'store'`
  - [x] `estore_ratio`: transactions where `channel == 'estore'` AND NOT `is_click_collect`
  - [x] `click_collect_ratio`: transactions where `is_click_collect == True`
  - [x] `dominant_channel`: argmax across the three ratios
- [x] Task 3 — Compute brand and store diversity (AC: 8–9)
  - [x] Join to `df_clean` to count nunique `Marque_Desc` per customer → `nb_unique_brands`
  - [x] Join to `df_clean` to count nunique `store_code_name` per customer → `nb_unique_stores`
- [x] Task 4 — Implement `compute_behavioral_features()` in `src/feature_engineer.py`
  - [x] Accepts `df_customers` AND `df_clean` (raw transactions needed for ratios)
  - [x] Returns `df_customers` with all new columns added
- [x] Task 5 — Add notebook section in `first-analysis.ipynb`
  - [x] Call `compute_behavioral_features(df_customers, df_clean)`
  - [x] Display sample and describe for ratio columns
  - [x] Markdown cell explaining each feature's business meaning

## Dev Notes

### Architecture Guardrails

**Module:** `src/feature_engineer.py` — add to existing module.  
**Notebook:** `01_eda.ipynb` — E1 section, after US-1.4 cells.

**Channel ratio computation pattern** (requires `df_clean`, not `df_customers`):
```python
def compute_behavioral_features(df_customers: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    total_txn = df_clean.groupby('anonymized_card_code')['anonymized_Ticket_ID'].nunique()
    
    store_txn = df_clean[df_clean['channel'] == 'store'].groupby(
        'anonymized_card_code')['anonymized_Ticket_ID'].nunique()
    estore_txn = df_clean[(df_clean['channel'] == 'estore') & (~df_clean['is_click_collect'])].groupby(
        'anonymized_card_code')['anonymized_Ticket_ID'].nunique()
    cc_txn = df_clean[df_clean['is_click_collect']].groupby(
        'anonymized_card_code')['anonymized_Ticket_ID'].nunique()
    
    df_customers['store_ratio'] = (store_txn / total_txn).fillna(0)
    df_customers['estore_ratio'] = (estore_txn / total_txn).fillna(0)
    df_customers['click_collect_ratio'] = (cc_txn / total_txn).fillna(0)
    ...
```

**dominant_channel:**
```python
channel_cols = ['store_ratio', 'estore_ratio', 'click_collect_ratio']
channel_labels = ['store', 'estore', 'click_collect']
df_customers['dominant_channel'] = df_customers[channel_cols].idxmax(axis=1).map(
    dict(zip(channel_cols, channel_labels))
)
```

**discount_rate edge case:**
```python
df_customers['discount_rate'] = (
    df_customers['total_discount_eur'] / df_customers['total_sales_eur']
).fillna(0).clip(upper=1.0)
```

### Previous Story Output (1-4)
- `df_customers` with RFM columns + all aggregated columns from US-1.3
- `df_clean` still available in notebook (never overwritten — immutable raw copy)

### Output of This Story
- `df_customers` with added columns: `avg_basket_size_eur`, `avg_units_per_basket`, `discount_rate`, `store_ratio`, `estore_ratio`, `click_collect_ratio`, `dominant_channel`, `nb_unique_brands`, `nb_unique_stores`

### References
- [epics — US-1.5](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — DataFrame Format Patterns](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
None

### Completion Notes List
- `compute_behavioral_features()` added to `src/feature_engineer.py` — accepts `df_customers` + `df_clean`, returns 9 new columns
- 14 unit tests in `tests/test_feature_engineer_stories_5_6_7.py` — all pass
- Notebook cells added to `first-analysis.ipynb` (Cell 7 + markdown)

### File List
Files modified:
- `src/feature_engineer.py` (added `compute_behavioral_features()`)
- `first-analysis.ipynb` (added US-1.5 section)
- `tests/test_feature_engineer_stories_5_6_7.py` (new — 14 tests for US-1.5)
