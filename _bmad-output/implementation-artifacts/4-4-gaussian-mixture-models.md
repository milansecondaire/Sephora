# Story 4.4: Gaussian Mixture Models

Status: done

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

- [x] Task 1 — Implement `run_gmm()` in `src/clustering.py` (AC: 1–2)
  - [x] Fit GMM with `n_components=k_optimal`, `covariance_type='diag'`, `random_state=RANDOM_STATE`
  - [x] Return labels (hard assignments via `.predict()`)
- [x] Task 2 — Implement `evaluate_gmm_bic_aic()` in `src/clustering.py` + BIC/AIC plot (AC: 3)
  - [x] Evaluate GMM for k in `range(2, 21)` (not full K_RANGE — cap for runtime)
  - [x] Plot BIC and AIC vs. k inline in notebook; save to `figures/gmm_bic_aic.png`
- [x] Task 3 — Compute metrics and add to comparison (AC: 4)
  - [x] Silhouette, DB, CH for `gmm_label` on `X_scaled`
  - [x] Append to `comparison_results` list
- [x] Task 4 — Log GMM run to MLflow (AC: 5)
  - [x] Wrap with `mlflow.start_run(run_name="gmm-final")`
  - [x] Log: `k`, `algorithm="gmm_diag"`, `n_features`, `silhouette`, `davies_bouldin`, `calinski_harabasz`, `min_cluster_pct`
- [x] Task 5 — Add notebook section in `02_clustering.ipynb`

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
Claude Opus 4.6

### Debug Log References
- mlflow dependency installed in anaconda env during test setup

### Completion Notes List
- ✅ `run_gmm()` implemented with `covariance_type='diag'`, `random_state=42`, `max_iter=200`
- ✅ `evaluate_gmm_bic_aic()` evaluates BIC/AIC for k=2..20 (default)
- ✅ `plot_gmm_bic_aic()` added to visualization.py — BIC+AIC dual curve with best-k highlight
- ✅ 18 new tests added (TestRunGmm: 9, TestEvaluateGmmBicAic: 6, TestPlotGmmBicAic: 3)
- ✅ Full test suite: 61/61 passed, 0 regressions
- ✅ Notebook cells added: markdown header, BIC/AIC plot, GMM fit + metrics, MLflow log, comparison table update
- ✅ `comparison_results` list pattern adopted for extensibility (US-4.5)

### File List
Files modified:
- `src/clustering.py` — added `run_gmm()`, `evaluate_gmm_bic_aic()`, import `GaussianMixture`
- `src/visualization.py` — added `plot_gmm_bic_aic()`
- `tests/test_clustering.py` — added TestRunGmm, TestEvaluateGmmBicAic, TestPlotGmmBicAic (18 tests)
- `02_clustering.ipynb` — added US-4.4 section (6 cells: md header, BIC/AIC, GMM fit, MLflow log, comparison update)
