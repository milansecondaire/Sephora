# Story 5.2: Per-Cluster KPI Computation

Status: ready-for-dev

## Story

As a Marketing Manager,
I want to see the same 10 KPIs computed per cluster,
so that I can understand each segment's behavior in absolute terms.

## Acceptance Criteria

1. All 10 KPIs from US-5.1 computed per `cluster_id`
2. Cluster size (n customers and %) added as first columns
3. Results displayed as a matrix: rows = clusters, columns = KPIs
4. Heatmap visualization of the KPI matrix (normalized per column) for quick pattern reading

## Tasks / Subtasks

- [ ] Task 1 — Implement `compute_cluster_kpis()` in `src/profiling.py` (AC: 1–3)
  - [ ] Group by `cluster_id`; aggregate `NUMERICAL_KPIS` with mean
  - [ ] Add `n_customers` and `pct_customers` columns
  - [ ] Return `cluster_kpis_df`
- [ ] Task 2 — Implement `plot_cluster_kpi_heatmap()` in `src/visualization.py` (AC: 4)
  - [ ] Normalize KPI matrix per column (min-max or z-score)
  - [ ] Render seaborn heatmap
  - [ ] Save to `figures/cluster_kpi_heatmap.png`
- [ ] Task 3 — Add notebook section in `03_profiling.ipynb`
  - [ ] Call `compute_cluster_kpis(df_customers)` and display matrix
  - [ ] Call `plot_cluster_kpi_heatmap(cluster_kpis_df)`

## Dev Notes

### Architecture Guardrails

**Module:** `src/profiling.py` + `src/visualization.py`.  
**Notebook:** `03_profiling.ipynb` — E5 section, after US-5.1.

**cluster_kpis function:**
```python
def compute_cluster_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean KPIs per cluster. Returns DataFrame: rows=clusters, cols=KPIs + size."""
    cluster_kpis = df.groupby('cluster_id')[NUMERICAL_KPIS].mean()
    cluster_kpis.insert(0, 'n_customers', df.groupby('cluster_id').size())
    cluster_kpis.insert(1, 'pct_customers', cluster_kpis['n_customers'] / len(df) * 100)
    return cluster_kpis
```

**Heatmap normalization** — use min-max per column to make cross-feature comparison visual:
```python
from sklearn.preprocessing import MinMaxScaler
normalized = MinMaxScaler().fit_transform(cluster_kpis[NUMERICAL_KPIS])
```

**Axes labeled in English** as per architecture output language requirement.

### Previous Story Output (5-1)
- `global_kpis` dict
- `df_customers` with `cluster_id`

### Output of This Story
- `cluster_kpis_df`: cluster × KPI matrix
- `figures/cluster_kpi_heatmap.png`

### References
- [epics — US-5.2](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
_To be filled by Dev Agent_

### Debug Log References

### Completion Notes List

### File List
Files to modify:
- `src/profiling.py` (add `compute_cluster_kpis()`)
- `src/visualization.py` (add `plot_cluster_kpi_heatmap()`)
- `03_profiling.ipynb` (add US-5.2 section)
