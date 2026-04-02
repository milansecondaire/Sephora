# Story 5.5: Statistical Validation of Segment Differences

Status: ready-for-dev

## Story

As a Data Scientist,
I want to statistically confirm that clusters are genuinely different,
so that the segmentation is not the result of random noise.

## Acceptance Criteria

1. Kruskal-Wallis test run on each of the 10 KPIs across all clusters
2. p-values tabulated; threshold p < 0.05
3. At least 7 of the 10 KPIs show significant differences (p < 0.05)
4. Post-hoc Dunn test (or Mann-Whitney pairwise) for KPIs that pass Kruskal-Wallis
5. Results summary: "X/10 KPIs significantly differ across segments at p < 0.05"

## Tasks / Subtasks

- [ ] Task 1 — Implement `run_kruskal_wallis()` in `src/validation.py` (AC: 1–3)
  - [ ] For each KPI in `NUMERICAL_KPIS`, run `kruskal` test across cluster groups
  - [ ] Return DataFrame: KPI | H-statistic | p-value | significant (bool)
- [ ] Task 2 — Post-hoc pairwise tests (AC: 4)
  - [ ] For each significant KPI: run pairwise Mann-Whitney U between all cluster pairs
  - [ ] Display significant pairs
- [ ] Task 3 — Print summary (AC: 5)
- [ ] Task 4 — Add notebook section in `03_profiling.ipynb`

## Dev Notes

### Architecture Guardrails

**Module:** `src/validation.py` — add to existing module (created in US-4.6).  
**Notebook:** `03_profiling.ipynb` — E5 section, after US-5.4.

**Kruskal-Wallis function:**
```python
from scipy.stats import kruskal

def run_kruskal_wallis(df: pd.DataFrame, kpis: list) -> pd.DataFrame:
    """Run Kruskal-Wallis test for each KPI across cluster groups."""
    results = []
    groups_by_cluster = {cid: grp for cid, grp in df.groupby('cluster_id')}
    for kpi in kpis:
        groups = [grp[kpi].dropna().values for grp in groups_by_cluster.values()]
        stat, pval = kruskal(*groups)
        results.append({'kpi': kpi, 'H_statistic': round(stat, 3), 
                        'p_value': pval, 'significant': pval < 0.05})
    return pd.DataFrame(results)
```

**Post-hoc suggestion:** Use `scipy.stats.mannwhitneyu` for pairwise, or install `scikit-posthocs` for Dunn test:
```python
# Pairwise Mann-Whitney for each significant KPI:
from scipy.stats import mannwhitneyu
from itertools import combinations
```

**Summary assertion:**
```python
n_significant = kw_results['significant'].sum()
print(f"Statistical validation: {n_significant}/{len(kpis)} KPIs significantly differ (p < 0.05)")
assert n_significant >= 7, f"Only {n_significant}/10 KPIs are significant — consider revisiting clustering"
```

### Previous Story Output (5-4)
- `df_customers` with `cluster_id`
- `NUMERICAL_KPIS` defined in `profiling.py`

### Output of This Story
- `kw_results_df`: Kruskal-Wallis results table
- Statistical validation summary printed

### References
- [epics — US-5.5](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — validation.py module](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
_To be filled by Dev Agent_

### Debug Log References

### Completion Notes List

### File List
Files to modify:
- `src/validation.py` (add `run_kruskal_wallis()`)
- `03_profiling.ipynb` (add US-5.5 section)
