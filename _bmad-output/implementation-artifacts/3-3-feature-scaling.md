# Story 3.3: Multi-Encoding & Scaling Pipeline (REVISED)

Status: complete

## Story

As a Data Scientist,
I want to apply the correct encoding per feature type and then scale all features uniformly,
so that categorical, ordinal, and continuous features all contribute properly to distance-based clustering.

## Acceptance Criteria

1. **Frequency encoding** applied to `customer_city` (~12K unique → 1 frequency column `customer_city_freq`)
2. **One-Hot encoding** applied to 9 low-cardinality categoricals → 37 dummy columns
3. **StandardScaler** applied to ALL resulting features (continuous + encoded)
4. Rationale documented: why StandardScaler after encoding (zero-mean, unit-variance for K-Means distances)
5. `X_scaled` shape printed: expected `(64469, 70)` — 30 continuous + 37 dummies + 2 missing indicators + 1 frequency
6. `preprocess_for_clustering()` single-function pipeline in `src/preprocessing.py`
7. No NaN, no Inf in final output

## Tasks / Subtasks

- [x] Task 1 — Implement encoding + scaling in `preprocess_for_clustering()` (AC: 1–7)
  - [x] Step 6: Frequency encode `customer_city` → `customer_city_freq`, drop original
  - [x] Step 7: One-Hot encode 9 categoricals via `pd.get_dummies(drop_first=False)`
  - [x] Step 8: Convert any remaining bool columns to int
  - [x] Step 9: Drop any remaining non-numeric columns (safety net)
  - [x] Step 10: StandardScaler on all 70 features
  - [x] Verify: mean ≈ 0, std ≈ 1, no NaN, no Inf
- [x] Task 2 — Update notebook `02_clustering.ipynb` E3.3 section (AC: 4, 5)
  - [x] Call `preprocess_for_clustering(df_customers)` to get `X_scaled`
  - [x] Print shape and verify
  - [x] Markdown cell: encoding decisions per type + StandardScaler rationale

## Dev Notes

### Architecture Guardrails

**Module:** `src/preprocessing.py` — all in `preprocess_for_clustering()`.
**Notebook:** `02_clustering.ipynb` — replaces old E3.2 + E3.3 sections.

**KEY CHANGE vs old story:** The old `scale_features()` applied StandardScaler to 21 numeric-only features. The new pipeline:
1. Frequency-encodes high-cardinality categoricals
2. One-Hot encodes low-cardinality categoricals
3. THEN StandardScaler on everything (including dummies — this is correct for K-Means distance)

**Encoding decisions:**

| Encoding | Applied to | Rationale |
|---|---|---|
| **Frequency** | `customer_city` (12K unique) | High cardinality → direct one-hot would create 12K sparse columns. Frequency = proxy for city size/urbanization |
| **One-Hot** | 9 features (3–6 categories each) | Low cardinality, no ordinal relationship (gender, channel, axe, market, country...) |
| **StandardScaler** | All 70 final features | Zero-mean, unit-variance required for K-Means (Euclidean distance). Applied AFTER encoding |

**One-Hot details:**
| Feature | Categories | Dummies |
|---|---|---|
| `gender` | Women, Men, Unknown | 3 |
| `dominant_channel` | store, estore, click_collect | 3 |
| `dominant_axe` | MAKE UP, SKINCARE, FRAGRANCE, HAIRCARE, OTHERS | 5 |
| `dominant_market` | SELECTIVE, EXCLUSIVE, SEPHORA, OTHERS | 4 |
| `country` | FR, LU, MC | 3 |
| `channel_recruitment` | store, estore, Unknown | 3 |
| `age_category` | 15-25yo, 26-35yo, 46-60yo, m60yo, Unknown | 5 |
| `age_generation` | genz, gena, geny, babyboomers, Unknown | 5 |
| `first_purchase_axe` | MAKE UP, SKINCARE, FRAGRANCE, HAIRCARE, OTHERS, Unknown | 6 |
| **Total** | | **37 dummies** |

**`drop_first=False` decision:** Keeping all dummies (not dropping one reference category) because StandardScaler will center them anyway, and for interpretability in cluster profiling we want all categories visible.

**Old functions retained:** `impute_features()` and `scale_features()` still work for backward compatibility (26 existing tests pass).

### Previous Story Output (3-2 revised)
- All features imputed, inf cleaned, missing indicators created

### Output of This Story
- `X_scaled`: fully preprocessed DataFrame (64469, 70), ready for PCA/clustering
- Old `scale_features()` still available for backward compatibility

### References
- [epics — US-3.3 (REVISED)](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Sonnet 4.6 (GitHub Copilot)

### Debug Log References
- Verified mean range: [-0.000000, 0.000000], std range: [1.000008, 1.000008] — correct StandardScaler behavior
- StandardScaler step was missing from `preprocess_for_clustering()` despite Task 1 being marked complete — added in this rework pass
- 36/36 tests pass (10 tests pour `preprocess_for_clustering()` ajoutés)

### Code Review Fixes (CR)
- **Fixed:** Ajout de `if "age" in df.columns` et `if "channel_recruitment" in df.columns` pour éviter les KeyError sur des sous-ensembles (Critic issue résolu).
- **Fixed:** Remplacement de `pd.get_dummies` par `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')` pour une meilleure robustesse en production vis-à-vis des catégories inconnues (Medium issue résolu).
- **Fixed:** Ajout du flag `return_transformers=True` renvoyant le pipeline d'imputation, OneHotEncoder, map fréquentiel et StandardScaler, fixant le manque de conservation d'artefacts ML.
- **Fixed:** Les assertions de tests `abs(...) < 1e-9` ont été remplacées par `np.testing.assert_allclose(...)` et `pytest.approx(...)`.
- **Fixed:** Ajout d'un test pour `_clean_primary_axe()` dans `test_clean_axe.py`.

### Completion Notes List
- **Code fixed:** Added StandardScaler (Step 10) to `preprocess_for_clustering()` — was previously missing despite Task 1 being marked complete
- **Notebook updated:** Cell `#VSC-4bde55c2` converted from broken code cell to proper Markdown cell (E3.3 header + encoding decisions table + StandardScaler rationale)
- **Notebook updated:** Cell `#VSC-33601b9d` replaced old `scale_features(X_imputed)` call with `X_scaled = preprocess_for_clustering(df_customers)` + shape assertion + mean/std verification
- Final shape: (64469, 70) — confirmed no NaN, no Inf

### File List
Files modified:
- `src/preprocessing.py` — added StandardScaler (Step 10) to `preprocess_for_clustering()`
- `tests/test_preprocessing.py` — added `TestPreprocessForClusteringOutputShape` + `TestPreprocessForClusteringScaling` (9 new tests)
- `02_clustering.ipynb` — E3.3 section updated: markdown cell + code cell
- `tests/test_transformers_output.py` — ajouté
- `tests/test_clean_axe.py` — ajouté
