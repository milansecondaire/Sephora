# Story 4.6: Cluster Stability Validation

Status: ready-for-dev

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

## Tasks / Subtasks

- [ ] Task 1 — Implement `bootstrap_stability()` in `src/validation.py` (AC: 1–2)
  - [ ] Re-run the best algorithm (selected in US-4.5) on 5×80% subsamples
  - [ ] Align labels to full-data labels and compute ARI
  - [ ] Return results DataFrame
- [ ] Task 2 — Evaluate stability threshold (AC: 3–5)
  - [ ] Compute mean ARI
  - [ ] Print pass/fail vs. 0.70 threshold
  - [ ] If fail: document and suggest remediation
- [ ] Task 3 — Create `src/validation.py` module
- [ ] Task 4 — Add notebook section in `02_clustering.ipynb` (E4 final section)

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

stability_df = bootstrap_stability(X_cluster, df_customers['final_cluster'], final_algo)
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

### References
- [epics — US-4.6](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — validation.py module boundary](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
_To be filled by Dev Agent_

### Debug Log References

### Completion Notes List

### File List
Files to create:
- `src/validation.py` (create — add `bootstrap_stability()`)

Files to modify:
- `02_clustering.ipynb` (add US-4.6 section + E4 summary)
