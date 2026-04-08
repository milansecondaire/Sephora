# Story 5.2: Per-Cluster KPI Computation

Status: done

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

- [x] Task 1 — Implement `compute_cluster_kpis()` in `src/profiling.py` (AC: 1–3)
  - [x] Group by `cluster_id`; aggregate `NUMERICAL_KPIS` with mean
  - [x] Add `n_customers` and `pct_customers` columns
  - [x] Return `cluster_kpis_df`
- [x] Task 2 — Implement `plot_cluster_kpi_heatmap()` in `src/visualization.py` (AC: 4)
  - [x] Normalize KPI matrix per column (min-max or z-score)
  - [x] Render seaborn heatmap
  - [x] Save to `figures/cluster_kpi_heatmap.png`
- [x] Task 3 — Add notebook section in `03_profiling.ipynb`
  - [x] Call `compute_cluster_kpis(df_customers)` and display matrix
  - [x] Call `plot_cluster_kpi_heatmap(cluster_kpis_df)`

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
Claude Opus 4.6

### Debug Log References
None — all tasks completed without issues.

### Completion Notes List
- Task 1: `compute_cluster_kpis()` added to `src/profiling.py` — groups by `cluster_id`, computes mean of `NUMERICAL_KPIS`, adds `n_customers` and `pct_customers` as first two columns. 10 unit tests added (TestComputeClusterKpis).
- Task 2: `plot_cluster_kpi_heatmap()` added to `src/visualization.py` — uses MinMaxScaler per column, renders seaborn heatmap with annotations, saves to `figures/`. 4 unit tests added (TestPlotClusterKpiHeatmap).
- Task 3: US-5.2 section added to `03_profiling.ipynb` — markdown header + compute_cluster_kpis display cell + plot_cluster_kpi_heatmap cell.
- Full regression: 356 passed, 1 pre-existing failure in test_config.py (unrelated).

### Review Follow-ups (AI)
- Fixed imports grouping in `src/visualization.py`.
- Optimized `df.groupby` in `src/profiling.py` function.
- Added default `save_path` value to `plot_cluster_kpi_heatmap`.

### File List
Files modified:
- `src/profiling.py` (added `compute_cluster_kpis()`)
- `src/visualization.py` (added `plot_cluster_kpi_heatmap()`)
- `tests/test_profiling.py` (added 10 tests for US-5.2)
- `tests/test_visualization.py` (added 4 tests for US-5.2)
- `03_profiling.ipynb` (added US-5.2 section: 3 cells)
