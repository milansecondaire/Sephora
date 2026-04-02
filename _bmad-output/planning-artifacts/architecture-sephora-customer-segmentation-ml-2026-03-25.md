---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7]
inputDocuments:
  - '_bmad-output/planning-artifacts/prd-sephora-customer-segmentation-ml-2026-03-25.md'
  - '_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md'
workflowType: 'architecture'
project_name: 'Edram'
user_name: 'Milan'
date: '2026-03-25'
version: '1.0'
status: 'Approved'
---

# Architecture Decision Document
## Advanced Customer Segmentation & Persona Creation — Sephora (Use Case 2)

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

---

## Project Context Analysis

### Requirements Overview

**Functional Requirements (8 categories):**

| Category | Architectural Implication |
|---|---|
| FR-1 — Data Preparation & Feature Engineering | Transaction-to-customer aggregation pipeline; missing value handling; normalization layer |
| FR-2 — Exploratory Data Analysis | Static visualization generation (distributions, heatmaps, 3D scatter); no real-time interaction required |
| FR-3 — Dimensionality Reduction | PCA (scikit-learn) + UMAP for 2D projection; visualization component decoupled from clustering |
| FR-4 — Clustering | Multi-algorithm loop (KMeans, Agglomerative, DBSCAN, GMM); k=[2..30] evaluation; best model selection |
| FR-5 — Cluster Validation | Metrics computation (Silhouette, DB, CH, Kruskal-Wallis/ANOVA); bootstrapping for stability |
| FR-6 — Segment Profiling | KPI computation per cluster + delta vs. global average; pivot table; CLV ranking |
| FR-7 — Persona Cards | Markdown + matplotlib/seaborn visual output; template defined in PRD Appendix B |
| FR-8 — Recommendations | Narrative content per segment; 2×2 priority matrix |

**Non-Functional Requirements (architectural impact):**

| NFR | Architectural Constraint |
|---|---|
| NFR-1 Reproducibility | Fixed `random_state` everywhere; versioned `requirements.txt` |
| NFR-2 Documentation | Jupyter notebooks with explanatory markdown cells for every analytical decision |
| NFR-3 Interpretability | Natural-language outputs for marketing — not only raw numbers |
| NFR-4 Performance | Full pipeline < 30 min on standard laptop; sampling for initial exploration if needed |
| NFR-5 Data Privacy | `anonymized_card_code` only; no external dataset cross-referencing |
| NFR-6 Modularity | Reusable functions/classes; clear module organization per pipeline stage |
| NFR-7 Visualization quality | Consistent matplotlib/seaborn palette; export ≥ 300 DPI |

**Epics & Stories — Architectural Mapping:**

| Epic | Stories | Architectural Component |
|---|---|---|
| E1 — Data Foundation | US-1.1 → US-1.7 | `data_loader.py` + `feature_engineer.py` |
| E2 — EDA | US-2.1 → US-2.6 | `01_eda.ipynb` + `visualization.py` |
| E3 — Feature Engineering & Preprocessing | US-3.1 → US-3.5 | `preprocessing.py` (scaling, PCA, UMAP) |
| E4 — Clustering Models | US-4.1 → US-4.6 | `02_clustering.ipynb` + `clustering.py` |
| E5 — Segment Profiling | US-5.1 → US-5.6 | `03_profiling.ipynb` + `profiling.py` |
| E6 — Persona Cards & Recommendations | US-6.1 → US-6.5 | `04_personas.ipynb` + `persona_generator.py` |

### Scale & Complexity

- **Primary domain:** Data Science / ML Analytics Pipeline (batch)
- **Complexity level:** Medium — multi-algorithm ML evaluation, no production deployment
- **Real-time requirements:** None — batch processing only
- **Multi-tenancy:** None
- **Regulatory compliance:** Low (anonymized data, academic context)
- **Integration complexity:** Low (local CSV input, file outputs)
- **Data volume:** Static 2025 dataset — full row count to be determined during EDA (US-1.3)
- **Estimated architectural components:** ~8 Python modules + ~4 Jupyter notebooks

### Technical Constraints & Dependencies

- Static local CSV dataset — no database connector, no external API
- Python 3.10+ on standard laptop environment
- No production deployment — academic/analytical context
- All code remains within a single local repository
- No additional variables available beyond the 34 CSV columns (Constraint C6 from PRD)

### Cross-Cutting Concerns Identified

1. **Reproducibility** — Global `random_state` constant to be passed to all stochastic components (KMeans, PCA, UMAP, bootstrapping, GMM)
2. **File path management** — Centralized path configuration for `data/`, `data/processed/`, `_bmad-output/`
3. **Analytical decision logging** — Every algorithmic choice documented inline in notebooks
4. **Visual palette consistency** — Shared color dictionary across all visualization modules
5. **Outlier handling strategy** — Unified global strategy to be defined before clustering (EDA phase gate)

---

## Technology Stack Decisions

### Primary Stack (Pre-selected in PRD Section 9.1 — confirmed)

| Layer | Library | Pinned Version |
|---|---|---|
| Language | Python | 3.11 (LTS; 3.10+ required per PRD) |
| Data manipulation | pandas | 2.2.x |
| Numerical computing | numpy | 1.26.x |
| Visualization (static) | matplotlib | 3.8.x |
| Visualization (statistical) | seaborn | 0.13.x |
| Visualization (interactive) | plotly | 5.20.x |
| ML / Dimensionality reduction | scikit-learn | 1.4.x |
| Non-linear dim. reduction | umap-learn | 0.5.x |
| Clustering algorithms | scikit-learn (KMeans, AgglomerativeClustering, DBSCAN, GaussianMixture) | 1.4.x |
| Cluster validation metrics | scikit-learn (silhouette_score, davies_bouldin_score, calinski_harabasz_score) | 1.4.x |
| Statistical tests | scipy.stats | 1.12.x |
| Notebook environment | JupyterLab | 4.x |

### Additional Tooling Decisions

| Tool | Decision | Rationale |
|---|---|---|
| Virtual environment | `venv` (stdlib) | Zero-overhead, no additional install; sufficient for single-developer academic project |
| Dependency management | `requirements.txt` (pip freeze) | Simple and reproducible; no over-engineering for academic context |
| t-SNE | `sklearn.manifold.TSNE` | Already in scikit-learn; no additional library needed alongside UMAP |
| Bootstrapping/subsampling | `numpy` random sampling | No dedicated library needed |
| Output export formats | `.csv` (data) + `.md` (reports) + `.png`/`.pdf` (figures) | Sufficient for academic deliverable |
| Random state constant | `RANDOM_STATE = 42` defined in `src/config.py` | Single source of truth for all stochastic components |

### No Starter Template

This project is a batch ML analytics pipeline, not a web application. There is no framework CLI to scaffold from. The project structure will be defined manually in Step 6 as a custom Python data science layout.

### Dependency File

A `requirements.txt` will be the sole dependency manifest, generated via `pip freeze` after environment setup. A companion `requirements-dev.txt` is not needed for this project scope.

---

## Core Architectural Decisions

### Category 1 — Data Architecture

**D1.1 — Intermediate Storage Strategy**

- **Decision:** CSV files in `data/processed/`
- **Files:** `customers_raw.csv` → `customers_features.csv` → `customers_clustered.csv`
- **Rationale:** Simple, inspectable, no additional dependencies; appropriate for academic batch context

**D1.2 — Missing Value Imputation Strategy**

| Field | Strategy | Rationale |
|---|---|---|
| `age = 0` or implausible low age | Flag `age_missing = True`; use `age_category` / `age_generation` as primary proxies | Confirmed: 0 = unknown per PRD |
| `gender = 99999` (not declared) | **Drop rows** — exclude from dataset | User decision: prefer clean dataset over imputation for this field |
| `customer_city` missing | Impute with `store_city`; if both missing → `"Unknown"` | Better than losing the row entirely |
| First-purchase fields missing | Create binary flag `has_first_purchase_data = False` | Preserves customer record while flagging data gap |

### Category 2 — ML Pipeline Architecture

**D2.1 — Centralized Configuration**

Single source of truth in `src/config.py`:
```python
RANDOM_STATE = 42
DATA_RAW_PATH = "data/BDD#7_Database_Albert_School_Sephora.csv"
DATA_PROCESSED_PATH = "data/processed/"
OUTPUT_PATH = "_bmad-output/implementation-artifacts/"
RECENCY_REFERENCE_DATE = "2025-12-31"
K_RANGE = range(2, 31)
```
All modules and notebooks import from this file — no hardcoded constants elsewhere.

**D2.2 — Multi-Algorithm Evaluation Pattern**

Dictionary-based instantiation in `src/clustering.py` with a single evaluation loop:
```python
CLUSTERING_ALGORITHMS = {
    "kmeans":      KMeans(n_clusters=k, random_state=RANDOM_STATE),
    "hierarchical": AgglomerativeClustering(n_clusters=k),
    "gmm":         GaussianMixture(n_components=k, random_state=RANDOM_STATE),
    "dbscan":      DBSCAN(eps=eps, min_samples=min_samples),
}
```
The loop iterates this dictionary and auto-generates the comparative metrics table.

**D2.3 — Feature Scaling**

- **Decision:** `StandardScaler` (scikit-learn)
- **Rationale:** Distance-based algorithms (KMeans, Hierarchical) require zero-mean, unit-variance features; more robust than MinMaxScaler for this feature distribution

**D2.4 — Outlier Handling Strategy**

- **Decision:** Retain all customers in the primary clustering
- **DBSCAN role:** Used specifically for noise/outlier detection, not as primary segmentation algorithm
- **Exclusion threshold:** Only customers flagged as noise by DBSCAN AND representing < 0.5% of the base are excluded
- **Rationale:** Extreme spenders / highly inactive customers may constitute valid marketing segments

### Category 3 — Output Architecture

**D3.1 — Persona Card Format**

- **Double format:** Markdown `.md` (report integration) + matplotlib figure (visual export)
- **Template:** Appendix B of PRD defines the card structure

**D3.2 — Output File Locations**

| Output | Path |
|---|---|
| Cleaned customer features | `data/processed/customers_features.csv` |
| Cluster assignments per customer | `data/processed/customers_clustered.csv` |
| Delta KPI table | `_bmad-output/implementation-artifacts/kpi_delta_table.md` |
| Persona cards (markdown) | `_bmad-output/implementation-artifacts/personas/segment_{n}.md` |
| Figures / charts | `_bmad-output/implementation-artifacts/figures/` |
| Notebooks (source) | project root |

### Decisions Summary

| # | Decision | Retained Choice |
|---|---|---|
| D1.1 | Intermediate storage | CSV in `data/processed/` |
| D1.2 | Missing values — gender=99999 | **Drop rows** |
| D1.2 | Missing values — age=0 | Flag + use age_category proxy |
| D1.2 | Missing values — customer_city | Impute with store_city or "Unknown" |
| D1.2 | Missing values — first purchase | Binary flag `has_first_purchase_data` |
| D2.1 | Centralized config | `src/config.py` |
| D2.2 | Multi-algorithm evaluation | Dictionary pattern + single loop |
| D2.3 | Feature scaling | `StandardScaler` |
| D2.4 | Outlier handling | Retain + DBSCAN noise detection; exclude if < 0.5% |
| D3.1 | Persona card format | `.md` + matplotlib figure |
| D3.2 | Output locations | `data/processed/` + `_bmad-output/implementation-artifacts/` |

---

## Implementation Patterns & Consistency Rules

### Naming Patterns

**Python variables & engineered columns**

| Convention | Rule | Example |
|---|---|---|
| Python variables | `snake_case` | `recency_days`, `monetary_avg` |
| Engineered CSV columns | `snake_case` with family prefix | `rfm_recency`, `channel_store_ratio`, `axis_skincare_share`, `socio_age_category` |
| Raw CSV columns | Preserve original name exactly | `anonymized_card_code`, `salesVatEUR` |
| Constants | `UPPER_SNAKE_CASE` | `RANDOM_STATE`, `K_RANGE` |
| Classes | `PascalCase` | `CustomerFeatureBuilder`, `ClusterEvaluator` |
| Functions | `snake_case` verb + subject | `load_raw_data()`, `compute_rfm_features()` |

**Files & modules**

| Convention | Rule | Example |
|---|---|---|
| Python modules | `snake_case.py` | `data_loader.py`, `feature_engineer.py` |
| Jupyter notebooks | `NN_description.ipynb` (2-digit prefix) | `01_eda.ipynb`, `02_clustering.ipynb` |
| Processed CSV outputs | `customers_{stage}.csv` | `customers_features.csv`, `customers_clustered.csv` |
| Markdown reports | `{type}_{identifier}.md` | `segment_1.md`, `kpi_delta_table.md` |
| Figure exports | `{type}_{description}.png` | `elbow_kmeans.png`, `umap_2d_clusters.png` |

---

### Structure Patterns

- Tests in `tests/` at root — **not** co-located with `src/` modules
- Each `src/foo.py` module has a corresponding `tests/test_foo.py`
- **Notebooks = orchestration + visualization + markdown narrative only** — no business logic inline
- **All logic lives in `src/`** — notebooks import exclusively from `src/`

---

### DataFrame Format Patterns

- Functions in `src/` always return `pd.DataFrame` — never bare `np.ndarray`
- Customer-level DataFrame index = `anonymized_card_code` (set as index, not column)
- Engineered feature columns always prefixed by family: `rfm_`, `channel_`, `axis_`, `socio_`, `lifecycle_`
- Cluster output column: always named `cluster_id` (int, 0-indexed)
- `customers_clustered.csv` always contains: `anonymized_card_code` + `cluster_id` + all feature columns
- Selected model name stored in `SELECTED_MODEL_NAME` in `src/config.py` after evaluation

---

### Error Handling Patterns

**In `src/` modules:**
- Data loading functions raise `FileNotFoundError` with full path in message
- Validation functions raise `ValueError` with descriptive message
- No silent `try/except` — always re-raise or log explicitly

**In notebooks — assertion gates after each critical step:**
```python
assert df_customers.index.name == "anonymized_card_code"
assert "cluster_id" in df_clustered.columns
assert df_clustered["cluster_id"].nunique() >= 2
```

---

### Visualization Patterns

**Shared color palette in `src/config.py`:**
```python
SEGMENT_COLORS = plt.cm.tab20.colors  # up to 20 distinct segments
PALETTE_AXES = {
    "SKINCARE":   "#FF6B8A",
    "MAKE UP":    "#A855F7",
    "FRAGRANCE":  "#F59E0B",
    "HAIRCARE":   "#10B981",
    "OTHERS":     "#6B7280",
}
```

**Figure standards:**
- Default `figsize`: `(12, 6)` for bar/line plots, `(10, 8)` for scatter/2D projections
- Export DPI: `300` for all `savefig()` calls
- Always call `plt.tight_layout()` before `savefig()`
- Titles and axis labels in **English** (document output language)

---

### Reproducibility Patterns

- `RANDOM_STATE = 42` defined once in `src/config.py` — **never hardcoded elsewhere**
- First cell of every notebook: `from src.config import *`
- `np.random.seed(RANDOM_STATE)` called at the start of every notebook

---

## Project Structure & Boundaries

### Complete Directory Tree

```
sephora-segmentation/                                   ← project root
│
├── README.md                                           ← project description + setup instructions
├── requirements.txt                                    ← pip freeze of Python environment
├── .gitignore                                          ← excludes data/processed/, __pycache__/
│
├── data/
│   ├── BDD#7_Database_Albert_School_Sephora.csv        ← raw data (READ ONLY — never modified)
│   └── processed/                                      ← pipeline outputs
│       ├── customers_raw.csv                           ← transaction→customer aggregation (E1)
│       ├── customers_features.csv                      ← engineered + scaled features (E1/E3)
│       └── customers_clustered.csv                     ← features + cluster_id (E4)
│
├── src/                                                ← reusable Python modules
│   ├── __init__.py
│   ├── config.py                                       ← RANDOM_STATE, paths, K_RANGE, palettes
│   ├── data_loader.py                                  ← load_raw_data(), validate_schema()
│   │                                                      (E1 / US-1.1)
│   ├── feature_engineer.py                             ← compute_rfm(), compute_channel_features(),
│   │                                                      compute_axis_features(), encode_categoricals()
│   │                                                      (E1 / US-1.2 → 1.6)
│   ├── preprocessing.py                                ← scale_features(), apply_pca(), apply_umap()
│   │                                                      (E3 / US-3.1 → 3.3)
│   ├── clustering.py                                   ← CLUSTERING_ALGORITHMS dict, evaluate_models(),
│   │                                                      select_best_model()  (E4 / US-3.4 → 3.7)
│   ├── validation.py                                   ← compute_silhouette(), compute_db_index(),
│   │                                                      run_kruskal_wallis(), bootstrap_stability()
│   │                                                      (E4 / FR-5)
│   ├── profiling.py                                    ← compute_global_kpis(), compute_cluster_kpis(),
│   │                                                      build_delta_table(), rank_by_clv()
│   │                                                      (E5 / US-4.1 → 4.5)
│   ├── persona_generator.py                            ← generate_persona_card(), render_persona_md()
│   │                                                      (E6 / US-5.1 → 5.2)
│   ├── recommendations.py                              ← build_marketing_recommendations(),
│   │                                                      build_priority_matrix()  (E6 / US-5.3 → 5.4)
│   └── visualization.py                                ← plot_distributions(), plot_correlation_heatmap(),
│                                                          plot_elbow(), plot_umap(), plot_cluster_profiles()
│                                                          (shared across all epics)
│
├── 01_eda.ipynb                                        ← E1 + E2: loading, cleaning, full EDA
├── 02_clustering.ipynb                                 ← E3 + E4: preprocessing, multi-algo, selection
├── 03_profiling.ipynb                                  ← E5: KPIs per cluster, delta table, CLV ranking
├── 04_personas.ipynb                                   ← E6: persona cards + marketing recommendations
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_feature_engineer.py
│   ├── test_preprocessing.py
│   ├── test_clustering.py
│   ├── test_validation.py
│   └── test_profiling.py
│
└── _bmad-output/
    └── implementation-artifacts/
        ├── kpi_delta_table.md
        ├── figures/
        │   ├── elbow_kmeans.png
        │   ├── silhouette_curve.png
        │   ├── umap_2d_clusters.png
        │   ├── cluster_profiles_radar.png
        │   └── priority_matrix.png
        └── personas/
            ├── segment_0.md
            ├── segment_1.md
            └── ...
```

### Module Boundaries & Responsibilities

| Module | Responsibility | Input Boundary | Output Boundary |
|---|---|---|---|
| `data_loader.py` | Load and validate raw CSV | File path string | `pd.DataFrame`, index=`anonymized_card_code` |
| `feature_engineer.py` | Aggregate transactions → customer-level, compute all features | Raw transaction DataFrame | `customers_features.csv` |
| `preprocessing.py` | Feature scaling, PCA, UMAP | `customers_features.csv` | Scaled matrix + PCA/UMAP components |
| `clustering.py` | Multi-algorithm evaluation, best model selection | Scaled matrix | `customers_clustered.csv` with `cluster_id` |
| `validation.py` | Quality metrics + statistical tests | Clustered DataFrame | Comparative metrics table |
| `profiling.py` | KPIs per cluster + delta vs. global average | Clustered DataFrame | `kpi_delta_table.md` + profiled DataFrame |
| `persona_generator.py` | Generate persona cards | Profiled DataFrame | `personas/segment_{n}.md` files |
| `recommendations.py` | Marketing actions per segment | Profiled DataFrame | Recommendations section + priority matrix |
| `visualization.py` | All figures | Various DataFrames | `.png` files in `figures/` |

### Integration Points & Data Flow

```
data/BDD#7_...csv  ──► data_loader ──► feature_engineer ──► preprocessing
                                                                    │
                                                                    ▼
                                                              clustering
                                                                    │
                                                    ┌───────────────┴──────────────┐
                                                    ▼                              ▼
                                               validation                      profiling
                                                    │                              │
                                                    │                    ┌─────────┴──────────┐
                                                    │                    ▼                    ▼
                                                    └──────► persona_generator    recommendations
                                                                         │                    │
                                                                         ▼                    ▼
                                                                   personas/*.md    recommendations output
```

All stages write their primary output to `data/processed/` or `_bmad-output/implementation-artifacts/`. Notebooks import from `src/` and orchestrate the flow — no cross-module imports between `src/` modules (each is independently callable).

---

## Architecture Validation

### Decision Coherence

| Check | Status |
|---|---|
| scikit-learn 1.4.x covers KMeans, AgglomerativeClustering, DBSCAN, GMM, StandardScaler, PCA, Silhouette/DB/CH scores | ✅ Single library, no conflicts |
| umap-learn 0.5.x compatible with numpy 1.26.x + scikit-learn 1.4.x | ✅ Compatible |
| pandas 2.2.x compatible with numpy 1.26.x | ✅ Compatible |
| plotly 5.20.x + matplotlib 3.8.x + seaborn 0.13.x — no conflicts | ✅ Compatible |
| `StandardScaler` appropriate for distance-based clustering (KMeans, Hierarchical) | ✅ Correct |
| `RANDOM_STATE = 42` declared in `config.py`, imported everywhere | ✅ Single source of truth |
| Dictionary pattern for clustering evaluation → single loop → compatible with all validation metrics | ✅ Coherent |
| Outlier retention + DBSCAN for noise detection → no contradiction with multi-algorithm strategy | ✅ Coherent |

### Requirements Coverage

| Requirement | Architectural Coverage | Status |
|---|---|---|
| FR-1 Data Preparation | `data_loader.py` + `feature_engineer.py` + D1.2 imputation rules | ✅ |
| FR-2 EDA | `01_eda.ipynb` + `visualization.py` | ✅ |
| FR-3 PCA + UMAP | `preprocessing.py` | ✅ |
| FR-4 Multi-algorithm Clustering | `clustering.py` (dict pattern, k=[2..30]) | ✅ |
| FR-5 Cluster Validation | `validation.py` | ✅ |
| FR-6 Segment Profiling + Delta Table | `profiling.py` | ✅ |
| FR-7 Persona Cards | `persona_generator.py` + dual format `.md`/figure | ✅ |
| FR-8 Marketing Recommendations | `recommendations.py` + priority matrix | ✅ |
| NFR-1 Reproducibility | Centralized `RANDOM_STATE`, `requirements.txt`, notebook seeds | ✅ |
| NFR-2 Documentation | Notebooks = orchestration + markdown narrative | ✅ |
| NFR-3 Interpretability | `.md` natural-language outputs per segment | ✅ |
| NFR-4 Performance < 30 min | Local batch, sampling available if needed | ✅ |
| NFR-5 Data Privacy | `anonymized_card_code` only; gender=99999 rows dropped | ✅ |
| NFR-6 Modularity | 8 independent `src/` modules + 4 orchestrating notebooks | ✅ |
| NFR-7 Visualization Quality | DPI 300, centralized palette, `tight_layout()` enforced | ✅ |

### Gap Analysis

| Severity | Gap | Resolution |
|---|---|---|
| ⚠️ Important | No structured logging strategy | `print()` statements sufficient for academic notebooks — no `logging` lib needed |
| ℹ️ Nice-to-have | No automated setup script / Makefile | Out of scope for academic project |
| ℹ️ Nice-to-have | No CI/CD pipeline | Out of scope — local project |

**No critical gaps identified. Architecture is implementation-ready.**

### Implementation Readiness Assessment

- ✅ All critical decisions documented with versions
- ✅ All implementation patterns comprehensive and conflict-preventing
- ✅ Complete file/directory structure defined and mapped to epics
- ✅ Integration points and data flow clearly specified
- ✅ All module boundaries well-defined with input/output contracts
- ✅ All NFRs architecturally addressed

---

## Handoff Summary

This architecture document is approved and ready for implementation handoff.

| Artifact | Location |
|---|---|
| PRD | `_bmad-output/planning-artifacts/prd-sephora-customer-segmentation-ml-2026-03-25.md` |
| Epics & Stories | `_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md` |
| Architecture (this document) | `_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md` |

**Next step:** Implementation — start with Epic 1 (Data Foundation): `data_loader.py` + `feature_engineer.py` + `01_eda.ipynb`.
