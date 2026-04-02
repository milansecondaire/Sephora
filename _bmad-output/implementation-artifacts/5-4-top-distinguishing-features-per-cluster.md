# Story 5.4: Top Distinguishing Features per Cluster

Status: ready-for-dev

## Story

As a Data Scientist,
I want to identify which features most strongly define each cluster,
so that persona narratives are grounded in statistical evidence.

## Acceptance Criteria

1. For each cluster: compute mean of each feature vs. rest-of-population mean
2. Rank features by absolute standardized difference (Cohen's d): `(cluster_mean - global_mean) / global_std`
3. Top 5 distinguishing features listed per cluster (positive = above average; negative = below)
4. Bar chart per cluster: top 5 features, x-axis = Cohen's d

## Tasks / Subtasks

- [ ] Task 1 — Implement `compute_distinguishing_features()` in `src/profiling.py` (AC: 1–3)
  - [ ] For each cluster: compute Cohen's d per feature
  - [ ] Return dict: cluster_id → sorted features DataFrame with Cohen's d
- [ ] Task 2 — Implement `plot_distinguishing_features()` in `src/visualization.py` (AC: 4)
  - [ ] One horizontal bar chart per cluster (top 5 pos + top 5 neg)
  - [ ] Save to `figures/distinguishing_features_cluster_{n}.png` per cluster
- [ ] Task 3 — Add notebook section in `03_profiling.ipynb`
  - [ ] Display top 5 features per cluster as tables
  - [ ] Display bar charts

## Dev Notes

### Architecture Guardrails

**Module:** `src/profiling.py` + `src/visualization.py`.  
**Notebook:** `03_profiling.ipynb` — E5 section, after US-5.3.

**Cohen's d computation:**
```python
def compute_distinguishing_features(df: pd.DataFrame, features: list) -> dict:
    """Return top distinguishing features per cluster via Cohen's d."""
    global_mean = df[features].mean()
    global_std = df[features].std()
    result = {}
    for cluster_id in df['cluster_id'].unique():
        cluster_df = df[df['cluster_id'] == cluster_id]
        cluster_mean = cluster_df[features].mean()
        cohens_d = (cluster_mean - global_mean) / (global_std + 1e-10)
        result[cluster_id] = cohens_d.abs().sort_values(ascending=False).reset_index()
        result[cluster_id].columns = ['feature', 'cohens_d_abs']
        result[cluster_id]['cohens_d'] = (cluster_mean - global_mean)[result[cluster_id]['feature'].values].values / (global_std[result[cluster_id]['feature'].values].values + 1e-10)
    return result
```

**This story's output directly feeds E6 persona writing** — the top distinguishing features are the foundation for persona name choices and narrative descriptions.

**Features to use:** `CLUSTERING_FEATURES` or `NUMERICAL_KPIS` — use the same feature set that was used for clustering.

### Previous Story Output (5-3)
- `delta_df` available
- `df_customers` with `cluster_id` and all features

### Output of This Story
- `distinguishing_features`: dict per cluster of top distinguishing features
- `figures/distinguishing_features_cluster_{n}.png` for each cluster

### References
- [epics — US-5.4](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
_To be filled by Dev Agent_

### Debug Log References

### Completion Notes List

### File List
Files to modify:
- `src/profiling.py` (add `compute_distinguishing_features()`)
- `src/visualization.py` (add `plot_distinguishing_features()`)
- `03_profiling.ipynb` (add US-5.4 section)
