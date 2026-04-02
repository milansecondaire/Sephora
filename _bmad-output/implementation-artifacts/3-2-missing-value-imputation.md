# Story 3.2: Imputation & Missing Indicators (REVISED)

Status: done

## Story

As a Data Scientist,
I want to handle missing values with type-appropriate strategies AND create missing indicators for high-missingness features,
so that missingness patterns become informative rather than destructive.

## Acceptance Criteria

1. Missing rate per feature printed
2. Continuous features with NaN → **median imputation** (age: 13.4%, subscription_tenure_days: 0.8%, first_purchase_amount: 62.7%)
3. Categorical features with NaN → **`'Unknown'` fill** (channel_recruitment: 62.7%, age_category: 30.5%, age_generation: 32.5%, first_purchase_axe: 62.7%, customer_city: 5.2%)
4. **Missing indicators created** before imputation: `has_age_info` (binary 0/1), `has_first_purchase_info` (binary 0/1)
5. Infinite values in ratio columns cleaned (inf/−inf → 0.0): 10 inf values across 7 ratio columns
6. `first_purchase_axe` cleaned: 2118 raw values simplified to 6 primary axes (MAKE UP, SKINCARE, FRAGRANCE, HAIRCARE, OTHERS, Unknown) — pipe-separated multi-label split + MAEK UP typo fix
7. Zero NaN confirmed in output

## Tasks / Subtasks

- [x] Task 1 — Implement imputation logic in `preprocess_for_clustering()` in `src/preprocessing.py` (AC: 1–7)
  - [x] Step 1: Clean `first_purchase_axe` via `_clean_primary_axe()` helper
  - [x] Step 2: Create `has_age_info` and `has_first_purchase_info` indicators
  - [x] Step 3: Drop 13 useless features (`FEATURES_DROP`)
  - [x] Step 4: Clean inf values in ratio columns → 0.0
  - [x] Step 5: Median impute continuous features (`FEATURES_CONTINUOUS`)
  - [x] Step 6: `'Unknown'` fill categorical features (`FEATURES_ONEHOT` + `FEATURES_FREQUENCY`)
  - [x] Assert zero NaN after imputation
- [x] Task 2 — Update notebook `02_clustering.ipynb` E3.2 section (AC: 1)
  - [x] Call `preprocess_for_clustering()` and display imputation summary
  - [x] Markdown cell: imputation decisions per feature type

## Dev Notes

### Architecture Guardrails

**Module:** `src/preprocessing.py` — integrated into `preprocess_for_clustering()`.
**KEY CHANGE vs old story:** The old `impute_features()` dropped features with >30% missing. The new approach KEEPS them — `first_purchase_amount` (62.7% NaN) is median-imputed + we create `has_first_purchase_info` to capture the missingness signal.

**Missing indicator rationale:**
- `age`: 13.4% missing — the fact that age is unknown may correlate with customer type (online-only, privacy-conscious)
- `first_purchase_*`: 62.7% missing — this is a systematic gap likely tied to channel_recruitment being unknown; the missingness itself is informative

**`_clean_primary_axe()` logic:**
```python
KNOWN_AXES = {"MAKE UP", "SKINCARE", "FRAGRANCE", "HAIRCARE", "OTHERS"}
def _clean_primary_axe(val):
    if pd.isna(val): return np.nan
    primary = str(val).split("|")[0].strip().replace("MAEK UP", "MAKE UP")
    return primary if primary in KNOWN_AXES else "OTHERS"
```

### Previous Story Output (3-1 revised)
- `FEATURES_DROP`, `FEATURES_CONTINUOUS`, `FEATURES_ONEHOT`, `FEATURES_FREQUENCY` in `src/config.py`

### Output of This Story
- All features imputed (zero NaN), infinite values cleaned, missing indicators created
- Old `impute_features()` still available for backward compatibility

### References
- [epics — US-3.2 (REVISED)](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6 (GitHub Copilot)

### Debug Log References
- `market_sephora_ratio` and `market_others_ratio` had inf values (division by zero upstream) — cleaned in Step 4

### Completion Notes List
- **Code done:** `preprocess_for_clustering()` in `src/preprocessing.py` handles all imputation with proper missing rate prints (AC-1) and strict zero NaN assertions (AC-7). Scaler has been removed to avoid scope-creep (now left to Story 3.3).
- **Notebook done:** E3.2 section updated — markdown + code cell call `preprocess_for_clustering()`
- Old `impute_features()` retained for backward compatibility (26 tests still pass)
- Tests written and passed in `tests/test_feature_engineer_clustering.py`
- Verified: age (8617 NaN → median 38.0), subscription_tenure_days (503 → median 2200.0), first_purchase_amount (40418 → median 55.97), channel_recruitment (40418 → Unknown), age_category (19638 → Unknown), age_generation (20937 → Unknown), first_purchase_axe (40418 → Unknown), customer_city (3371 → Unknown)

### File List
Files modified:
- `src/preprocessing.py` — added `_clean_primary_axe()`, imputation steps in `preprocess_for_clustering()`, removed StandardScaler.
- `02_clustering.ipynb` — E3.2 markdown cell updated (new imputation table + key changes section); E3.2 code cell updated to call `preprocess_for_clustering()` with missing-rate summary display
- `tests/test_feature_engineer_clustering.py` — created tests for `preprocess_for_clustering` and `_clean_primary_axe`.
