# Story 2.4: Channel, Product & Brand Analysis

Status: review

## Story

As a Marketing Manager,
I want to understand channel mix and product affinities at the population level,
so that I can set marketing expectations for segment output.

## Acceptance Criteria

1. Pie/bar chart: store vs. estore vs. Click & Collect share (global)
2. Stacked bar: product axis mix by channel
3. Stacked bar: market tier mix (SELECTIVE / EXCLUSIVE / SEPHORA) by loyalty status
4. Top 20 brands by total sales EUR (horizontal bar chart)
5. Sales EUR by `age_generation` and `gender`
6. Monthly transaction volume chart (seasonality check)

## Tasks / Subtasks

- [x] Task 1 — Implement `plot_channel_product_overview()` in `src/visualization.py` (AC: 1–3)
  - [x] Bar chart: `store_ratio`, `estore_ratio`, `click_collect_ratio` global means
  - [x] Stacked bar of axis ratios by `dominant_channel`
  - [x] Stacked bar of market ratios by `loyalty_status`
  - [x] Save to `figures/channel_product_overview.png`
- [x] Task 2 — Compute top 20 brands (AC: 4)
  - [x] Load `df_clean` or use `df_customers` — note: brand data is transaction-level
  - [x] Group by brand, sum `salesVatEUR`, take top 20
  - [x] Horizontal bar chart saved to `figures/top20_brands.png`
- [x] Task 3 — Sales by demographics (AC: 5)
  - [x] Cross-tab `age_generation` × `gender` with `monetary_total` mean
  - [x] Heatmap or grouped bar chart
  - [x] Save to `figures/sales_by_demographics.png`
- [x] Task 4 — Monthly transaction volume (AC: 6)
  - [x] Requires transaction-level data (`df_clean`) — group by month of `transactionDate`
  - [x] Line chart saved to `figures/monthly_volume.png`
- [x] Task 5 — Add notebook section in `first-analysis.ipynb`

## Dev Notes

### Architecture Guardrails

**Module:** `src/visualization.py` — add to existing module.  
**Notebook:** `01_eda.ipynb` — E2 section, after US-2.3.

**Requires `df_clean`:** AC items 4 and 6 need transaction-level data. Ensure `df_clean` is kept in scope in notebook (do not overwrite it after E1).

**Top 20 brands:** Column for brand in raw CSV is `Marque_Desc`. Verify column name matches exactly.

**Monthly volume:**
```python
df_clean['month'] = df_clean['transactionDate'].dt.to_period('M')
monthly = df_clean.groupby('month').size()
monthly.index = monthly.index.to_timestamp()
```

**Stacked bar axis mix by channel — from `df_customers`:**
```python
axe_cols = ['axe_make_up_ratio', 'axe_skincare_ratio', 'axe_fragrance_ratio', 
            'axe_haircare_ratio', 'axe_others_ratio']
df_customers.groupby('dominant_channel')[axe_cols].mean().plot(kind='bar', stacked=True)
```

### Previous Story Output (2-3)
- `df_customers` and `df_clean` available in notebook

### Output of This Story
- `figures/channel_product_overview.png`
- `figures/top20_brands.png`
- `figures/sales_by_demographics.png`
- `figures/monthly_volume.png`

### References
- [epics — US-2.4](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
No issues encountered.

### Completion Notes List
- Added `plot_channel_product_overview()`, `plot_top_brands()`, `plot_sales_by_demographics()`, `plot_monthly_volume()` to `src/visualization.py`
- Tests: 8 tests across 4 test classes — all passing
- Notebook cells E2-4a/b/c/d added for all 4 charts

### File List
Files modified:
- `src/visualization.py` (added 4 functions)
- `tests/test_visualization.py` (added 8 tests)
- `first-analysis.ipynb` (added US-2.4 section)
