# Story 1.2: Data Quality Assessment & Cleaning

Status: done

## Story

As a Data Analyst,
I want to identify and fix all known data quality issues,
so that downstream features are computed on reliable, consistent data.

## Acceptance Criteria

1. Missing value count and rate per column displayed (`df.isnull().sum()`)
2. Typo corrected: `Axe_Desc == 'MAEK UP'` → `'MAKE UP'`
3. `age == 0` or age < 15 flagged as missing (`np.nan`) — do not drop the row
4. `gender == 99999` replaced with `'Unknown'`
5. `gender` recoded: `1 → 'Men'`, `2 → 'Women'`
6. `status` recoded: `1 → 'No Fid'`, `2 → 'BRONZE'`, `3 → 'SILVER'`, `4 → 'GOLD'`
7. Click & Collect orders identified: `channel == 'estore'` AND `store_type_app` not in estore-type values → add column `is_click_collect = True`
8. A Markdown cell summarizes all transformations applied and the number of rows affected by each

## Tasks / Subtasks

- [x] Task 1 — Implement `assess_data_quality()` in `src/feature_engineer.py` (AC: 1)
  - [x] Print missing values per column with count and % rate
  - [x] Return a summary dict (used in notebook display)
- [x] Task 2 — Implement `clean_raw_data()` in `src/feature_engineer.py` (AC: 2, 3, 4, 5, 6, 7)
  - [x] Create `df_clean = df.copy()` — never mutate the original frame
  - [x] Fix `Axe_Desc` typo: `'MAEK UP'` → `'MAKE UP'`
  - [x] Flag `age == 0` or `age < 15` as `np.nan`
  - [x] Replace `gender == 99999` with `'Unknown'`; recode 1→'Men', 2→'Women'
  - [x] Recode `status`: 1→'No Fid', 2→'BRONZE', 3→'SILVER', 4→'GOLD'
  - [x] Add `is_click_collect` boolean column
  - [x] Return `df_clean`
- [x] Task 3 — Add notebook section in `first-analysis.ipynb` (AC: 1, 8)
  - [x] Call `assess_data_quality(df)` and display summary
  - [x] Call `clean_raw_data(df)` to get `df_clean`
  - [x] Add Markdown cell listing every transformation with rows affected

## Dev Notes

### Architecture Guardrails

**Module:** `src/feature_engineer.py` — create if it doesn't exist yet.  
**Notebook:** `01_eda.ipynb` — continuation of E1 section after US-1.1 cells.

**Function signatures:**
```python
# src/feature_engineer.py
import pandas as pd
import numpy as np

def assess_data_quality(df: pd.DataFrame) -> dict:
    """Print and return a summary of missing values and known quality issues."""
    ...

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning rules. Returns a NEW DataFrame (df_clean).
    Never modifies the input df in place.
    """
    ...
```

**Click & Collect rule (from architecture D1.2 and epics):**
```python
cc_mask = (df['channel'] == 'estore') & (
    ~df['store_type_app'].isin(['ESTORE', 'WEB', 'MOBILE', 'APP', 'CSC'])
)
df_clean['is_click_collect'] = cc_mask
```

**Age flagging rule:**
```python
df_clean.loc[(df_clean['age'] == 0) | (df_clean['age'] < 15), 'age'] = np.nan
```

**Architecture decision D1.2 — gender=99999:**
> Replace with 'Unknown' — do NOT drop the row (confirmed in architecture doc).
> `gender == 99999` rows are kept — only `gender==99999` treated as unknown, not excluded.

### Previous Story Output (1-1)
- `src/config.py` exists with `DATA_RAW_PATH`
- `src/data_loader.py` exists with `load_raw_data()` returning `df`
- Variable `df` (raw transaction DataFrame) available in notebook

### Output of This Story
- `df_clean`: cleaned transaction-level DataFrame (still transaction-level, one row per transaction)
- Available in notebook for US-1.3 (aggregation)

### Notebook Pattern
```python
# Cell — Data Quality Assessment
from src.feature_engineer import assess_data_quality, clean_raw_data
quality_summary = assess_data_quality(df)
df_clean = clean_raw_data(df)
print(f"Rows before: {len(df):,} | After: {len(df_clean):,}")
```

```markdown
<!-- Markdown cell -->
## Data Cleaning Summary
| Transformation | Rows Affected |
|---|---|
| Axe_Desc typo fixed (MAEK UP → MAKE UP) | X |
| age flagged as NaN (age=0 or <15) | X |
| gender=99999 → 'Unknown' | X |
| gender recoded (1/2 → Men/Women) | X |
| status recoded (1/2/3/4 → labels) | X |
| is_click_collect flagged | X |
```

### References
- [epics — US-1.2](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — D1.2 Missing Value Strategy](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — Module Boundaries: feature_engineer.py](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
No debug issues encountered.

### Completion Notes List
- Task 1: `assess_data_quality()` implemented — prints missing values per column with count/rate, returns summary dict. 17 columns have missing values detected.
- Task 2: `clean_raw_data()` implemented — all 6 cleaning rules applied: Axe_Desc typo fix (115,679 rows), age NaN flagging (34,657 rows), gender recode (99999→Unknown + 1→Men/2→Women), status recode (2/3/4→labels), is_click_collect boolean (12 rows). Zero rows dropped.
- Task 3: Three notebook cells added to `first-analysis.ipynb`: quality assessment call, cleaning call with transformation counts, and markdown summary table.
- 18 unit tests written covering both functions (4 for assess_data_quality, 14 for clean_raw_data). All 33 tests pass (15 existing + 18 new).
- Note: No `status=1` (No Fid) exists in the dataset — all statuses are 2, 3, or 4. Mapping still includes it for safety.
- Note: Notebook was `first-analysis.ipynb` (not `01_eda.ipynb` as originally specified in story — adapted to existing project structure from story 1-1).

### File List
Files created/modified:
- `src/feature_engineer.py` (created — `assess_data_quality`, `clean_raw_data`)
- `tests/test_feature_engineer.py` (created — 18 unit tests)
- `first-analysis.ipynb` (modified — added 3 cells for US-1.2 section)
