# Story 4.4: Gaussian Mixture Models

Status: ready-for-dev

## Story

As a Data Scientist,
I want to run a GMM (soft clustering) and compare it to hard clustering methods,
so that I can assess whether probabilistic assignment improves interpretability.

## Acceptance Criteria

1. `GaussianMixture` with `n_components = k_optimal`, `covariance_type='full'`, `random_state=42`
2. Cluster labels stored as `df_customers['gmm_label']`
3. BIC and AIC curves plotted over k range (additional model selection tools for GMM)
4. Silhouette, DB, CH scores added to comparison table (US-4.5)

## Tasks / Subtasks

- [ ] Task 1 — Implement `run_gmm()` in `src/clustering.py` (AC: 1–2)
  - [ ] Fit GMM with `n_components=k_optimal`, `covariance_type='full'`, `random_state=RANDOM_STATE`
  - [ ] Return labels (hard assignments via `.predict()`)
- [ ] Task 2 — BIC/AIC curves (AC: 3)
  - [ ] Evaluate GMM for k in a subset range (e.g., range(2, min(31, len(K_RANGE)+2)) — can reuse K_RANGE)
  - [ ] Plot BIC and AIC vs. k; save to `figures/gmm_bic_aic.png`
- [ ] Task 3 — Compute metrics and add to comparison (AC: 4)
  - [ ] Silhouette, DB, CH for `gmm_label`
  - [ ] Append to `comparison_results` list
- [ ] Task 4 — Add notebook section in `02_clustering.ipynb`

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

**Performance note:** GMM with `covariance_type='full'` is slow on large datasets with many features. If runtime > 10 min on full data, use `covariance_type='diag'` as fallback and document it. Alternatively, run on `X_pca` components only.

**BIC lower = better** — minimum BIC identifies optimal k for GMM.

### Previous Story Output (4-3)
- `df_customers['hclust_label']`
- `comparison_results` list started
- `k_optimal`

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
