# Story 1.6: Product Affinity Feature Computation

Status: complete

## Story

As a Data Analyst,
I want to compute product axis and market tier share features per customer,
so that the clustering captures product preferences.

## Acceptance Criteria

1. For each of `['MAKE UP', 'SKINCARE', 'FRAGRANCE', 'HAIRCARE', 'OTHERS']`:
   `axe_{name}_ratio` = share of `salesVatEUR` in that axis / `total_sales_eur` (lowercase name: `axe_make_up_ratio`, etc.)
2. For each of `['SELECTIVE', 'EXCLUSIVE', 'SEPHORA', 'OTHERS']`:
   `market_{name}_ratio` = share of `salesVatEUR` in that market (lowercase: `market_selective_ratio`, etc.)
3. `dominant_axe` = axis with highest spend share
4. `dominant_market` = market tier with highest spend share
5. `axis_diversity` = number of distinct axes purchased (1–5)
6. All ratio columns sum to 1.0 ± 0.001 per customer (assert checked)

## Tasks / Subtasks

- [x] Task 1 — Implement `compute_product_affinity_features()` in `src/feature_engineer.py` (AC: 1–6)
  - [x] Compute `salesVatEUR` per customer per `Axe_Desc` → pivot → divide by `total_sales_eur`
  - [x] Map axis names to column names (spaces/special chars → underscores, lowercase)
  - [x] Do the same for `Market_Desc` → market ratios
  - [x] Compute `dominant_axe` via `idxmax`; compute `dominant_market` via `idxmax`
  - [x] Compute `axis_diversity` = nunique `Axe_Desc` per customer
  - [x] Assert sum of axis ratios ≈ 1.0 per customer (use `np.testing.assert_allclose`)
  - [x] Return `df_customers` with all new columns
- [x] Task 2 — Add notebook section in `first-analysis.ipynb`
  - [x] Call `compute_product_affinity_features(df_customers, df_clean)`
  - [x] Display distribution of `dominant_axe` and `dominant_market`
  - [x] Markdown cell: business interpretation of product affinity features

## Dev Notes

### Architecture Guardrails

**Module:** `src/feature_engineer.py` — add to existing module.  
**Notebook:** `01_eda.ipynb` — E1 section, after US-1.5 cells.

**Axis column name mapping:**
```python
AXIS_NAMES = ['MAKE UP', 'SKINCARE', 'FRAGRANCE', 'HAIRCARE', 'OTHERS']
AXIS_COL_MAP = {name: f"axe_{name.replace(' ', '_').lower()}_ratio" for name in AXIS_NAMES}
# → {'MAKE UP': 'axe_make_up_ratio', 'SKINCARE': 'axe_skincare_ratio', ...}

MARKET_NAMES = ['SELECTIVE', 'EXCLUSIVE', 'SEPHORA', 'OTHERS']
MARKET_COL_MAP = {name: f"market_{name.lower()}_ratio" for name in MARKET_NAMES}
```

**Ratio computation pattern — pivot approach:**
```python
axis_sales = (
    df_clean.groupby(['anonymized_card_code', 'Axe_Desc'])['salesVatEUR']
    .sum()
    .unstack(fill_value=0)
)
# Normalize by total_sales_eur (join to df_customers)
axis_ratios = axis_sales.div(df_customers['total_sales_eur'], axis=0).fillna(0)
# Rename columns
axis_ratios.rename(columns=AXIS_COL_MAP, inplace=True)
df_customers = df_customers.join(axis_ratios[list(AXIS_COL_MAP.values())])
```

**Sum assertion:**
```python
axe_cols = list(AXIS_COL_MAP.values())
row_sums = df_customers[axe_cols].sum(axis=1)
np.testing.assert_allclose(row_sums, 1.0, atol=0.001, 
    err_msg="Product axis ratios do not sum to 1.0 per customer")
```

**Important:** The typo in `Axe_Desc` (`'MAEK UP'`) was already corrected in US-1.2 (`clean_raw_data`). The pivot should use `df_clean` which already has `'MAKE UP'` correct.

**axis_diversity:**
```python
df_customers['axis_diversity'] = df_clean.groupby('anonymized_card_code')['Axe_Desc'].nunique()
```

### Previous Story Output (1-5)
- `df_customers` with behavioral + channel + RFM features
- `df_clean` still available (includes `Axe_Desc`, `Market_Desc`, `salesVatEUR` per line)

### Output of This Story
- `df_customers` with added columns: `axe_make_up_ratio`, `axe_skincare_ratio`, `axe_fragrance_ratio`, `axe_haircare_ratio`, `axe_others_ratio`, `market_selective_ratio`, `market_exclusive_ratio`, `market_sephora_ratio`, `market_others_ratio`, `dominant_axe`, `dominant_market`, `axis_diversity`

### References
- [epics — US-1.6](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — Naming Patterns (column prefix conventions)](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
None

### Completion Notes List
- `compute_product_affinity_features()` added — pivot-based axis/market ratios with assertion gate
- Module-level constants: `AXIS_NAMES`, `AXIS_COL_MAP`, `MARKET_NAMES`, `MARKET_COL_MAP`
- 10 unit tests in `tests/test_feature_engineer_stories_5_6_7.py` — all pass
- Notebook cells added to `first-analysis.ipynb` (Cell 8 + markdown)

### File List
Files modified:
- `src/feature_engineer.py` (added `compute_product_affinity_features()`, constants)
- `first-analysis.ipynb` (added US-1.6 section)
- `tests/test_feature_engineer_stories_5_6_7.py` (10 tests for US-1.6)
