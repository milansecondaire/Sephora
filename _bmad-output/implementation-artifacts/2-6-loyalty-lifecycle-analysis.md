# Story 2.6: Loyalty & Lifecycle Analysis

Status: done

## Story

As a CRM Manager,
I want to understand how loyalty tiers correlate with purchase behavior,
so that I can interpret future segment loyalty distributions.

## Acceptance Criteria

1. Box plots of `monetary_avg`, `frequency`, `recency_days` by `loyalty_status` (No Fid / BRONZE / SILVER / GOLD)
2. Proportion of customers per tier printed
3. Mean `subscription_tenure_days` per status
4. `is_new_customer` rate per loyalty tier
5. Key insight summarized in Markdown: e.g., "GOLD customers spend Nx more on average than No Fid"

## Tasks / Subtasks

- [x] Task 1 — Implement `plot_loyalty_lifecycle()` in `src/visualization.py` (AC: 1)
  - [x] Box plots for `monetary_avg`, `frequency`, `recency_days` grouped by `loyalty_status`
  - [x] Order groups: No Fid < BRONZE < SILVER < GOLD
  - [x] Save to `figures/loyalty_lifecycle.png`
- [x] Task 2 — Compute loyalty tier summaries (AC: 2, 3, 4)
  - [x] `df_customers.groupby('loyalty_status').agg(...)` with all requested metrics
  - [x] Print proportion per tier
  - [x] Display as table
- [x] Task 3 — Add notebook section in `first-analysis.ipynb` (AC: 5)
  - [x] Call functions
  - [x] Markdown cell with key quantified insights + E2 summary gate statement

## Dev Notes

### Architecture Guardrails

**Module:** `src/visualization.py` — add to existing module.  
**Notebook:** `01_eda.ipynb` — E2 final section. After this story, E2 is complete.

**Loyalty status order for plots:**
```python
LOYALTY_ORDER = ['No Fid', 'BRONZE', 'SILVER', 'GOLD']
# Use as `order=LOYALTY_ORDER` in seaborn boxplot
```

**Box plot pattern:**
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, ['monetary_avg', 'frequency', 'recency_days']):
    sns.boxplot(data=df, x='loyalty_status', y=col, order=LOYALTY_ORDER, ax=ax)
    ax.set_title(f'{col} by Loyalty Status')
```

**E2 final Markdown summary:** After all E2 stories, add a summary Markdown cell consolidating:
- Key distribution insights
- Correlation decisions (from US-2.2)
- Outlier strategy confirmed (from US-2.5)
- Loyalty tier behavioral differences (from this story)
- **Gate statement:** "EDA complete — proceeding to E3 Feature Engineering"

### Previous Story Output (2-5)
- `df_customers` with `is_outlier` and all prior features

### Output of This Story
- `figures/loyalty_lifecycle.png`
- E2 analysis complete — `01_eda.ipynb` fully implemented

### References
- [epics — US-2.6](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6 (Amelia Dev Agent)

### Debug Log References
None — all tasks completed without errors.

### Completion Notes List
- `plot_loyalty_lifecycle()`: box plots of monetary_avg, frequency, recency_days by loyalty_status (ordered No Fid < BRONZE < SILVER < GOLD)
- `compute_loyalty_summary()`: returns aggregated stats per loyalty tier with proportion
- Key finding: No "No Fid" customers in aggregated data — all belong to BRONZE (66.4%), SILVER (29.6%), GOLD (4.0%)
- GOLD customers are most engaged: highest frequency, lowest recency, longest tenure
- 7 tests added and passing for US 2-6 functions
- E2 summary markdown cell added with gate statement

### File List
Files modified:
- `src/visualization.py` (added `plot_loyalty_lifecycle()`, `compute_loyalty_summary()`, `LOYALTY_ORDER`)
- `first-analysis.ipynb` (added US-2.6 section: cells E2-6a, E2-6b + insights markdown + E2 summary)
- `tests/test_visualization.py` (added TestPlotLoyaltyLifecycle, TestComputeLoyaltySummary)
