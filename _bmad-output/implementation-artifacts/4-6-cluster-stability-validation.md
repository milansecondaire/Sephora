# Story 4.6: Cluster Stability Validation

Status: done

> **Post-refonte R1 note:** No structural changes needed. `X_cluster` throughout is now `X_scaled` (43 features).
> `bootstrap_stability()` receives `X_scaled` directly — no dimension issue.
> **Post-refonte R1.5 note:** MLflow logging added for consistency with R1.5.
> Stability results (mean ARI, min/max ARI) logged as run `"stability-validation"` with bootstrap CSV as artifact.

## Story

As a Data Scientist,
I want to test the stability of the final clustering on data subsamples,
so that I can confirm segments are not artifacts of the exact sample.

## Acceptance Criteria

1. Run the final algorithm on 5 bootstrapped 80% subsamples
2. ARI (Adjusted Rand Index) computed between each subsample result and full-data result
3. Mean ARI ≥ 0.70 (acceptable stability threshold)
4. Results reported in a table: subsample # | ARI score | cluster count
5. If ARI < 0.70: document the instability and revisit k or algorithm
6. The stability validation run is logged in MLflow as a run named `"stability-validation"` with params (`n_bootstraps`, `subsample_frac`, `algorithm`, `k`) and metrics (`mean_ari`, `min_ari`, `max_ari`), and the bootstrap results CSV as artifact

## Tasks / Subtasks

- [x] Task 1 — Implement `bootstrap_stability()` in `src/validation.py` (AC: 1–2)
  - [x] Re-run the best algorithm (selected in US-4.5) on 5×80% subsamples
  - [x] Align labels to full-data labels and compute ARI
  - [x] Return results DataFrame
- [x] Task 2 — Evaluate stability threshold (AC: 3–5)
  - [x] Compute mean ARI
  - [x] Print pass/fail vs. 0.70 threshold
  - [x] If fail: document and suggest remediation
- [x] Task 3 — Create `src/validation.py` module
- [x] Task 4 — Log stability results to MLflow (AC: 6)
  - [x] Wrap with `mlflow.start_run(run_name="stability-validation")`
  - [x] Log params: `n_bootstraps`, `subsample_frac`, `algorithm` (name of chosen algo), `k` (`k_optimal`)
  - [x] Log metrics: `mean_ari`, `min_ari`, `max_ari`
  - [x] Save `stability_df` as CSV, log as MLflow artifact
- [x] Task 5 — Add notebook section in `02_clustering.ipynb` (E4 final section)

## Dev Notes

### Architecture Guardrails

**Module:** `src/validation.py` — create this module.  
**Notebook:** `02_clustering.ipynb` — E4 final section. After this story, E4 is complete.

**Module skeleton:**
```python
# src/validation.py
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score
from src.config import RANDOM_STATE

def bootstrap_stability(
    X: pd.DataFrame, 
    full_labels: pd.Series,
    algorithm_fn,          # callable: takes X and returns labels array
    n_bootstraps: int = 5,
    subsample_frac: float = 0.80,
) -> pd.DataFrame:
    """Run algorithm on N subsamples and compute ARI vs. full-data labels.
    
    algorithm_fn: a function that takes a DataFrame X and returns cluster labels array.
    """
    results = []
    for i in range(n_bootstraps):
        seed = RANDOM_STATE + i
        idx = X.sample(frac=subsample_frac, random_state=seed).index
        X_sub = X.loc[idx]
        sub_labels = algorithm_fn(X_sub)
        # ARI: compare full_labels for the subsample customers
        ari = adjusted_rand_score(full_labels.loc[idx], sub_labels)
        results.append({
            'bootstrap': i + 1, 
            'n_samples': len(idx),
            'ari': round(ari, 4),
            'n_clusters': len(set(sub_labels)),
        })
    return pd.DataFrame(results)
```

**algorithm_fn usage in notebook:**
```python
# Wrap the chosen algorithm as a callable
def final_algo(X):
    return run_kmeans_final(X, k_optimal)[0]  # or run_hierarchical etc.

stability_df = bootstrap_stability(X_scaled, df_customers['final_cluster'], final_algo)
print(f"Mean ARI: {stability_df['ari'].mean():.3f}")
```

**ARI threshold:**
```python
mean_ari = stability_df['ari'].mean()
if mean_ari >= 0.70:
    print(f"✓ Stable clustering (Mean ARI = {mean_ari:.3f} ≥ 0.70)")
else:
    print(f"⚠ Unstable clustering (Mean ARI = {mean_ari:.3f} < 0.70)")
    print("Consider: reduce k, try different algorithm, or review feature selection")
```

**MLflow logging:**
```python
import mlflow

mean_ari = stability_df['ari'].mean()
min_ari = stability_df['ari'].min()
max_ari = stability_df['ari'].max()

with mlflow.start_run(run_name="stability-validation"):
    mlflow.log_params({
        "n_bootstraps": 5,
        "subsample_frac": 0.80,
        "algorithm": best_algo_name,  # e.g. "kmeans"
        "k": k_optimal,
    })
    mlflow.log_metrics({
        "mean_ari": mean_ari,
        "min_ari": min_ari,
        "max_ari": max_ari,
    })
    stability_csv = "data/processed/stability_results.csv"
    stability_df.to_csv(stability_csv, index=False)
    mlflow.log_artifact(stability_csv, artifact_path="validation")
```

**E4 final Markdown summary:** Add after all validation:
- Best algorithm and k chosen
- Stability assessment result
- `final_cluster` column confirmed on `df_customers`
- **Gate statement:** "Clustering complete — proceeding to E5 Segment Profiling"

### Previous Story Output (4-5)
- `df_customers['final_cluster']` / `cluster_id`
- `X_cluster` available
- `k_optimal`, best algorithm type known

### Output of This Story (E4 Final Output)
- `stability_df`: bootstrap stability results
- `src/validation.py` created
- `02_clustering.ipynb` E4 section complete
- MLflow run `"stability-validation"` with ARI metrics + bootstrap CSV artifact

### References
- [epics — US-4.6](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — validation.py module boundary](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
No issues encountered.

### Completion Notes List
- Created `src/validation.py` with `bootstrap_stability()` function: runs N bootstrap subsamples, computes ARI vs full-data labels
- 12 unit tests in `tests/test_validation.py` — all passing (covers AC 1-5: DataFrame structure, ARI range, stability threshold on clear clusters, determinism, custom params)
- Notebook section 10 added: bootstrap stability run, threshold evaluation, MLflow logging, E4 gate statement
- MLflow run `stability-validation` logs params (n_bootstraps, subsample_frac, algorithm, k), metrics (mean_ari, min_ari, max_ari), and CSV artifact

### File List
Files created:
- `src/validation.py`
- `tests/test_validation.py`

Files modified:
- `02_clustering.ipynb` (added US-4.6 section 10 + E4 summary)
