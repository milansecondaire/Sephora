# Story 4.4: Gaussian Mixture Models

Status: ready-for-dev

> **Post-refonte R1 note:** `covariance_type` changed from `'full'` to `'diag'` — mandatory.
> With 43 features, `'full'` requires estimating 43×43 covariance matrices per cluster,
> which is numerically unstable and extremely slow at 64K rows. `'diag'` is the correct choice.
> All references to `X_cluster` / `X_pca` replaced by `X_scaled` (43 features).
> MLflow logging added for consistency with R1.5.

## Story

As a Data Scientist,
I want to run a GMM (soft clustering) and compare it to hard clustering methods,
so that I can assess whether probabilistic assignment improves interpretability.

## Acceptance Criteria

1. `GaussianMixture` with `n_components = k_optimal`, `covariance_type='diag'`, `random_state=42` — **NOT `'full'`** (43 features → too slow and numerically unstable)
2. Cluster labels stored as `df_customers['gmm_label']`
3. BIC and AIC curves plotted for k = 2..20 (not full K_RANGE — GMM is slower)
4. Silhouette, DB, CH scores added to comparison table (US-4.5)
5. The GMM run is logged in MLflow as a run named `"gmm-final"` with the same params/metrics schema as KMeans and Hierarchical (see R1.5)

## Tasks / Subtasks

- [ ] Task 1 — Implement `run_gmm()` in `src/clustering.py` (AC: 1–2)
  - [ ] Fit GMM with `n_components=k_optimal`, `covariance_type='diag'`, `random_state=RANDOM_STATE`
  - [ ] Return labels (hard assignments via `.predict()`)
- [ ] Task 2 — Implement `evaluate_gmm_bic_aic()` in `src/clustering.py` + BIC/AIC plot (AC: 3)
  - [ ] Evaluate GMM for k in `range(2, 21)` (not full K_RANGE — cap for runtime)
  - [ ] Plot BIC and AIC vs. k inline in notebook; save to `figures/gmm_bic_aic.png`
- [ ] Task 3 — Compute metrics and add to comparison (AC: 4)
  - [ ] Silhouette, DB, CH for `gmm_label` on `X_scaled`
  - [ ] Append to `comparison_results` list
- [ ] Task 4 — Log GMM run to MLflow (AC: 5)
  - [ ] Wrap with `mlflow.start_run(run_name="gmm-final")`
  - [ ] Log: `k`, `algorithm="gmm_diag"`, `n_features`, `silhouette`, `davies_bouldin`, `calinski_harabasz`, `min_cluster_pct`
- [ ] Task 5 — Add notebook section in `02_clustering.ipynb`

## Dev Notes

### Architecture Guardrails

**Module:** `src/clustering.py` — add to existing module.  
**Notebook:** `02_clustering.ipynb` — E4 section, after US-4.3.

**GMM function:**
```python
from sklearn.mixture import GaussianMixture

def run_gmm(X: pd.DataFrame, k: int) -> tuple:
    """Fit GMM. Returns (hard labels, fitted model)."""
    gmm = GaussianMixture(n_components=k, covariance_type='full', 
                          random_state=RANDOM_STATE, max_iter=200)
    labels = gmm.fit_predict(X)
    return labels, gmm
```

**BIC/AIC evaluation:**
```python
def evaluate_gmm_bic_aic(X: pd.DataFrame, k_range=None) -> pd.DataFrame:
    """Evaluate BIC and AIC for a range of k."""
    k_range = k_range or range(2, 21)  # keep small range for runtime
    results = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type='full', 
                              random_state=RANDOM_STATE, max_iter=200)
        gmm.fit(X)
        results.append({'k': k, 'bic': gmm.bic(X), 'aic': gmm.aic(X)})
    return pd.DataFrame(results)
```

**`covariance_type='diag'` is mandatory** — 43 features × k clusters × 43×43 matrices = infeasible with `'full'`.
With `'diag'`, each cluster has an independent diagonal covariance (one variance per feature). This is a valid and common choice for high-dimensional GMMs.

**BIC lower = better** — minimum BIC identifies optimal k for GMM.

**BIC/AIC range capped at k=20** — GMM is slower than KMeans. Range(2,21) is sufficient to see the minimum.

### Previous Story Output (4-3)
- `df_customers['hclust_label']`
- `comparison_results` list started (entries for KMeans and Hierarchical already there)
- `k_optimal`
- `X_scaled` shape (64469, 43) available — use this directly, no `X_pca`

### Output of This Story
- `df_customers['gmm_label']`
- `figures/gmm_bic_aic.png`
- GMM scores appended to `comparison_results`

### References
- [epics — US-4.4](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — D2.2 Dictionary Pattern for CLUSTERING_ALGORITHMS](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
_To be filled by Dev Agent_

### Debug Log References

### Completion Notes List

### File List
Files to modify:
- `src/clustering.py` (add `run_gmm()`, `evaluate_gmm_bic_aic()`)
- `src/visualization.py` (add `plot_gmm_bic_aic()`)
- `02_clustering.ipynb` (add US-4.4 section)
