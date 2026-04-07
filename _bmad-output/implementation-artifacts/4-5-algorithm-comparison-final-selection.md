# Story 4.5: Algorithm Comparison & Final Selection

Status: review

> **Post-refonte R1 note:** No structural changes needed. References to `X_cluster` replaced by `X_scaled`.
> The export to `customers_with_clusters.csv` is already specified in R1.6 Task 10 — Amelia must
> ensure both are consistent (same file, not duplicated). MLflow: log the comparison table as a CSV artifact.

## Story

As a Data Scientist,
I want to compare all algorithms on a common metrics table,
so that the final model choice is transparent and justified.

## Acceptance Criteria

1. Comparison table: Algorithm | k | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Min cluster size % | Notes
2. Best algorithm identified based on combined score
3. Chosen model's cluster labels stored as `df_customers['final_cluster']`
4. Markdown cell: 3–5 sentences justifying the choice
5. `df_customers` exported to `data/processed/customers_with_clusters.csv`

## Tasks / Subtasks

- [x] Task 1 — Build comparison table from `comparison_results` (AC: 1)
  - [x] Convert `comparison_results` list to DataFrame
  - [x] Add min cluster size % column
  - [x] Display as formatted table
- [x] Task 2 — Select best algorithm (AC: 2, 3)
  - [x] Identify algorithm with best combined score (highest silhouette, lowest DB, non-trivial cluster sizes)
  - [x] Assign `df_customers['final_cluster'] = df_customers[f'{best_algo}_label']`
  - [x] Also add `cluster_id` alias (architecture pattern): `df_customers['cluster_id'] = df_customers['final_cluster']`
- [x] Task 3 — Export `customers_with_clusters.csv` (AC: 5)
  - [x] Save `df_customers` with all features + `final_cluster` / `cluster_id` to `data/processed/customers_with_clusters.csv`
  - [x] ⚠️ This is the same file targeted by R1.6 Task 10 — do NOT create a second export cell; use the one already scaffolded by R1.6 and fill in `df_customers['final_cluster']` correctly before it runs
- [x] Task 4 — Log comparison table to MLflow (new after R1.5)
  - [x] Save `comp_df` as CSV to a temp file, log as MLflow artifact under run `"algorithm-comparison"`
  - [x] `mlflow.log_artifact("data/processed/comparison_results.csv", artifact_path="comparison")`
- [x] Task 5 — Add notebook section in `02_clustering.ipynb` (AC: 4)
  - [x] Display comparison table
  - [x] Markdown justification cell (Milan fills in)

## Dev Notes

### Architecture Guardrails

**Notebook:** `02_clustering.ipynb` — E4 section, after US-4.4.  
**No new module needed** — this is orchestration logic in the notebook.

**`cluster_id` naming (from architecture):**
> Cluster output column: always named `cluster_id` (int, 0-indexed).
> `customers_clustered.csv` always contains: `anonymized_card_code` + `cluster_id` + all feature columns.

```python
df_customers['cluster_id'] = df_customers['final_cluster']
df_customers.to_csv(DATA_PROCESSED_PATH + "customers_with_clusters.csv")
```

**Min cluster size % column:**
```python
for row in comparison_results:
    labels = df_customers[f"{row['algorithm'].lower()}_label"]
    row['min_cluster_pct'] = labels.value_counts(normalize=True).min() * 100
```

**Selection logic (semi-automated, documented):**
```python
comp_df = pd.DataFrame(comparison_results)
# Normalize: higher silhouette better, lower DB better
comp_df['score'] = comp_df['silhouette'] - comp_df['davies_bouldin'] / comp_df['davies_bouldin'].max()
best_algo_row = comp_df.loc[comp_df['score'].idxmax()]
print(f"Best algorithm: {best_algo_row['algorithm']}")
```

**Architecture note:** Even if GMM has best metrics, hard assignment (`.predict()`) is used in `final_cluster` to keep downstream logic simple.

### Previous Story Output (4-4)
- `comparison_results` with KMeans, Hierarchical, GMM entries
- `df_customers` with `kmeans_label`, `hclust_label`, `gmm_label`

### Output of This Story
- `df_customers['final_cluster']` and `df_customers['cluster_id']`
- `data/processed/customers_with_clusters.csv`
- Comparison table displayed in notebook

### References
- [epics — US-4.5](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — D3.2 Output File Locations](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — DataFrame Format Patterns (cluster_id naming)](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6 via GitHub Copilot

### Debug Log References
None — clean implementation, all tests passed first run.

### Completion Notes List
- Added `build_comparison_table()` to `src/clustering.py` — converts comparison_results list to DataFrame with min_cluster_pct column
- Added `select_best_algorithm()` to `src/clustering.py` — combined score: silhouette - DB/DB_max, penalty for <1% clusters
- 10 new tests in `tests/test_clustering.py` (TestBuildComparisonTable: 6, TestSelectBestAlgorithm: 4)
- Updated notebook cell #VSC-bb297f08: uses `build_comparison_table()` + `select_best_algorithm()` to display comparison table and pick best algo
- Added new MLflow logging cell: saves `comparison_results.csv` as artifact under `algorithm-comparison` run
- Updated `final_cluster` assignment cell: dynamically maps best_algo → label column (was hardcoded to kmeans_label)
- Added `cluster_id` alias column alongside `final_cluster`
- Updated export cell: assert `final_cluster` and `cluster_id` exist before export
- Added US-4.5 markdown section with justification placeholder for Milan
- 322/322 tests passing, 0 regressions

### File List
Files to create:
- `data/processed/customers_with_clusters.csv` (generated at notebook runtime)
- `data/processed/comparison_results.csv` (generated at notebook runtime)

Files modified:
- `src/clustering.py` — added `build_comparison_table()`, `select_best_algorithm()`
- `tests/test_clustering.py` — added TestBuildComparisonTable (6 tests), TestSelectBestAlgorithm (4 tests)
- `02_clustering.ipynb` — updated cells for US-4.5 (comparison table, MLflow, final_cluster assignment, export, markdown section)
