# Story 4.2: K-Means — Final Run & Assignment

Status: done

## Story

As a Data Scientist,
I want to run the final K-Means with the chosen k,
so that each customer receives a cluster label.

## Acceptance Criteria

1. Final K-Means run with `k = k_optimal` (chosen from US-4.1)
2. Cluster labels stored as `df_customers['kmeans_label']`
3. Cluster size distribution printed (count and % per cluster)
4. No cluster smaller than 0.5% of the base (flag if so)
5. Silhouette Score, Davies-Bouldin, Calinski-Harabasz printed for this final model

## Tasks / Subtasks

- [x] Task 1 — Implement `run_kmeans_final()` in `src/clustering.py` (AC: 1–5)
  - [x] Fit KMeans with `k_optimal`, fixed seed/params
  - [x] Return (labels array, fitted KMeans model)
- [x] Task 2 — Add notebook section in `02_clustering.ipynb`
  - [x] Assign `df_customers['kmeans_label']`
  - [x] Print cluster size distribution
  - [x] Flag clusters < 0.5% of base
  - [x] Print final 3 metrics

## Dev Notes

### Architecture Guardrails

**Module:** `src/clustering.py` — add to existing module.  
**Notebook:** `02_clustering.ipynb` — E4 section, after US-4.1.

**Function:**
```python
def run_kmeans_final(X: pd.DataFrame, k: int) -> tuple:
    """Run final KMeans with given k. Returns (labels, fitted_model)."""
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
    labels = km.fit_predict(X)
    return labels, km
```

**Cluster size check:**
```python
cluster_sizes = pd.Series(labels).value_counts(normalize=True)
min_size = cluster_sizes.min()
if min_size < 0.005:
    print(f"⚠ WARNING: Cluster {cluster_sizes.idxmin()} has only {min_size*100:.2f}% of customers")
```

**`df_customers` assignment:**
```python
labels, km_model = run_kmeans_final(X_cluster, k_optimal)
df_customers['kmeans_label'] = labels
```

Note: `df_customers.index` and `X_cluster.index` must be aligned (same index). They should be from the same source — verify with `assert (df_customers.index == X_cluster.index).all()`.

### Previous Story Output (4-1)
- `kmeans_metrics_df`, `k_optimal` chosen and stored in notebook
- `X_cluster` available

### Output of This Story
- `df_customers['kmeans_label']` — KMeans cluster assignments

### References
- [epics — US-4.2](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6 (GitHub Copilot)

### Debug Log References
- Kernel reload needed for notebook (cached module import)

### Completion Notes List
- Task 1: Implemented `run_kmeans_final(X, k)` → returns `(labels, fitted_model)` with `RANDOM_STATE=42`, `n_init=10`, `max_iter=300`. 7 unit tests added covering: return types, label count, determinism, model fit, dtype, different k.
- Task 2: Added US-4.2 markdown + code cell in notebook after US-4.1 summary. Cell covers AC 1–5: final run, label assignment, cluster size distribution (count + %), <0.5% flag (Cluster 4 flagged at 0.01%), and 3 quality metrics (Silhouette=0.1679, DB=1.4354, CH=6567.1).
- All 24 clustering tests pass. 1 pre-existing failure in `test_preprocessing.py::test_default_50_features_caps_at_30` (PCA, unrelated).
- **Code Review Fixes Applied:**
  - *High*: Reduced `silhouette_score` computation time in the notebook by adding `sample_size=10000` (down from ~19s to ~2s).
  - *Medium*: `df_customers` is now exported to `data/processed/customers_features_clustered.csv` after labels are assigned to ensure labels persist for US-4.3+
  - *Medium*: Replaced tautological `test_deterministic_with_random_state` test and added `test_kmeans_initialization_params` to mock the initialization footprint.
  - *Low*: Removed `importlib.reload` dev workaround from the final notebook cell and added `random_state` as an optional parameter to `run_kmeans_final`.

### File List
- `src/clustering.py` — added `run_kmeans_final()`
- `tests/test_clustering.py` — added `TestRunKmeansFinal` (7 tests)
- `02_clustering.ipynb` — added US-4.2 markdown + code cell
