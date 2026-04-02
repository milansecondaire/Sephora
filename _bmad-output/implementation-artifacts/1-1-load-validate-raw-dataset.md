# Story 1.1: Load & Validate Raw Dataset

Status: done

## Story

As a Data Analyst,
I want to load the raw CSV file and perform an initial quality check,
so that I know exactly what I'm working with before any transformation.

## Acceptance Criteria

1. File loads without encoding error — use `encoding='utf-8-sig'`
2. Total row count and column count displayed (expected: 34 columns)
3. Each column's dtype inferred and printed via `df.dtypes`
4. `anonymized_card_code` parsed as **string** (not float) to avoid precision loss from scientific notation
5. `transactionDate` and `first_purchase_dt` parsed as `datetime` objects
6. Sample of 5 rows displayed for visual inspection
7. A Markdown cell immediately follows the loading cell summarizing: total rows, total customers, date range, and any immediate observations

## Tasks / Subtasks

- [x] Task 1 — Create project scaffolding (AC: none — prerequisite)
  - [x] Create `src/` package with `__init__.py`
  - [x] Create `src/config.py` with all constants (see Dev Notes)
  - [x] Create `data/processed/` directory
  - [x] Create `src/data_loader.py` with `load_raw_data()` and `validate_schema()` functions
- [x] Task 2 — Implement `load_raw_data()` in `src/data_loader.py` (AC: 1, 3, 4, 5)
  - [x] Read CSV with correct `dtype` overrides and `parse_dates`
  - [x] Raise `FileNotFoundError` with full path if file not found
  - [x] Return a `pd.DataFrame` (not `np.ndarray`)
- [x] Task 3 — Implement `validate_schema()` in `src/data_loader.py` (AC: 2, 3, 4, 5)
  - [x] Check 34 columns present — raise `ValueError` with column diff if mismatch
  - [x] Verify `anonymized_card_code` dtype is `object` (str)
  - [x] Verify `transactionDate` and `first_purchase_dt` are `datetime64`
  - [x] Print validation summary
- [x] Task 4 — Implement notebook cell in `first-analysis.ipynb` (AC: 2, 3, 5, 6, 7)
  - [x] Import from `src.config` and `src.data_loader`
  - [x] Call `load_raw_data()` and `validate_schema()`
  - [x] Display `df.shape`, `df.dtypes`, `df.head(5)`
  - [x] Add Markdown cell summarizing initial observations

## Dev Notes

### Architecture Guardrails — MUST FOLLOW

**Project structure (from architecture doc):**
- All business logic goes in `src/` modules — NOT inline in notebooks
- Notebooks import exclusively from `src/` — zero ad-hoc logic in notebook cells
- Customer-level DataFrame index = `anonymized_card_code` (set as index, not column) — **but only after aggregation in US-1.3; here at load-time it stays as a column**
- `RANDOM_STATE = 42` defined ONCE in `src/config.py` — never hardcoded elsewhere
- First cell of every notebook: `from src.config import *` + `np.random.seed(RANDOM_STATE)`

**`src/config.py` content to create (complete, canonical):**
```python
# src/config.py
import matplotlib.pyplot as plt

RANDOM_STATE = 42

# File paths
DATA_RAW_PATH = "data/BDD#7_Database_Albert_School_Sephora.csv"
DATA_PROCESSED_PATH = "data/processed/"
OUTPUT_PATH = "_bmad-output/implementation-artifacts/"

# Dates
RECENCY_REFERENCE_DATE = "2025-12-31"

# Clustering
K_RANGE = range(2, 31)

# Visualization
SEGMENT_COLORS = plt.cm.tab20.colors  # up to 20 distinct segments
PALETTE_AXES = {
    "SKINCARE":   "#FF6B8A",
    "MAKE UP":    "#A855F7",
    "FRAGRANCE":  "#F59E0B",
    "HAIRCARE":   "#10B981",
    "OTHERS":     "#6B7280",
}

# Figure defaults
FIGSIZE_BAR = (12, 6)
FIGSIZE_SCATTER = (10, 8)
FIGURE_DPI = 300
```

**`src/data_loader.py` — function signatures and docstrings:**
```python
import pandas as pd
from src.config import DATA_RAW_PATH

EXPECTED_COLUMNS = 34
STRING_COLUMNS = ["anonymized_card_code", "anonymized_Ticket_ID", "anonymized_first_purchase_id"]
DATE_COLUMNS = ["transactionDate", "first_purchase_dt", "subscription_date"]

def load_raw_data(path: str = DATA_RAW_PATH) -> pd.DataFrame:
    """Load the Sephora raw transaction CSV with correct dtypes and date parsing.
    
    Raises FileNotFoundError if path does not exist.
    Returns a pd.DataFrame — never modifies the source file.
    """
    ...

def validate_schema(df: pd.DataFrame) -> None:
    """Validate that df matches the expected schema.
    
    Raises ValueError with a descriptive message if:
    - Column count != EXPECTED_COLUMNS
    - anonymized_card_code is not dtype object (str)
    - transactionDate / first_purchase_dt are not datetime64
    
    Prints a summary on success.
    """
    ...
```

### Naming Conventions (from architecture doc)

| Item | Convention | Example |
|---|---|---|
| Python functions | `snake_case` verb + subject | `load_raw_data()`, `validate_schema()` |
| Python modules | `snake_case.py` | `data_loader.py`, `config.py` |
| Constants | `UPPER_SNAKE_CASE` | `RANDOM_STATE`, `DATA_RAW_PATH` |
| Raw CSV columns | Preserve exactly | `anonymized_card_code`, `salesVatEUR` |

### Data Loading — Exact Parameters

```python
pd.read_csv(
    path,
    encoding="utf-8-sig",
    dtype={
        "anonymized_card_code": str,
        "anonymized_Ticket_ID": str,
        "anonymized_first_purchase_id": str
    },
    parse_dates=["transactionDate", "first_purchase_dt", "subscription_date"]
)
```

**Why `utf-8-sig`:** The CSV has a BOM (Byte Order Mark). Using `utf-8` would fail or produce a garbled first column name.

**Why string dtype for card codes:** These IDs contain scientific notation when parsed as float (e.g., `1.23e+15`), which loses precision. Force `str` at load time.

### Existing File Context

A script `_analysis_d1.py` already exists at project root with exploratory analysis code covering US-1.1 topics. Use it as a **reference only** — do NOT copy it into `src/`. The actual implementation in `src/data_loader.py` must follow the architecture patterns above (no print statements, functions return values, raise exceptions rather than printing errors).

Key insights from the existing analysis:
- The CSV has **34 columns** confirmed
- `anonymized_card_code` would parse as `float64` without the `dtype` override (scientific notation issue confirmed)
- `subscription_date` should also be in `parse_dates` (not just `transactionDate` and `first_purchase_dt`)

### Error Handling Pattern (from architecture doc)

```python
# CORRECT pattern for src/ modules:
if not Path(path).exists():
    raise FileNotFoundError(f"Raw data file not found: {path}")

# WRONG — do not use silent try/except:
try:
    df = pd.read_csv(path)
except Exception:
    return None  # ← NEVER do this
```

### Notebook Integration Pattern

The notebook `first-analysis.ipynb` already exists at project root. The implementation for this story adds the first structured notebook section. The first cells must follow this pattern:

```python
# Cell 1 — Setup (always first cell of every notebook)
from src.config import *
import numpy as np
np.random.seed(RANDOM_STATE)

# Cell 2 — Load & Validate
from src.data_loader import load_raw_data, validate_schema
df = load_raw_data()
validate_schema(df)
print(f"Shape: {df.shape}")
print(df.dtypes)
display(df.head(5))
```

```markdown
<!-- Markdown cell immediately after: -->
## Initial Dataset Observations
- **Total rows:** X,XXX,XXX transactions
- **Unique customers:** XXX,XXX
- **Date range:** YYYY-MM-DD → YYYY-MM-DD
- **34 columns** confirmed; no schema violations detected
- `anonymized_card_code` correctly typed as string
- Date columns parsed as datetime64
```

### Project Structure Notes

The `src/` package does NOT yet exist — this story creates it. Required directory tree to create:

```
sephora-segmentation/  (project root = /Users/milanviallet/Documents - MacBook Air de Milan/Albert/B2/BDD/Sephora)
├── src/
│   ├── __init__.py          ← empty file (marks package)
│   ├── config.py            ← centralized constants (CREATE THIS STORY)
│   └── data_loader.py       ← load_raw_data(), validate_schema() (CREATE THIS STORY)
├── data/
│   ├── BDD#7_Database_Albert_School_Sephora.csv   ← READ ONLY — never modify
│   └── processed/           ← CREATE THIS DIRECTORY (empty for now)
└── first-analysis.ipynb     ← EXISTS — add E1-S1.1 section
```

**Do NOT create yet:** `feature_engineer.py`, `preprocessing.py`, `clustering.py`, etc. Those belong to later stories.

### References

- [epics-sephora-customer-segmentation-ml-2026-03-25.md — US-1.1](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture-sephora-customer-segmentation-ml-2026-03-25.md — Technology Stack](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture-sephora-customer-segmentation-ml-2026-03-25.md — D2.1 Centralized Configuration](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture-sephora-customer-segmentation-ml-2026-03-25.md — Project Structure (Complete Directory Tree)](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture-sephora-customer-segmentation-ml-2026-03-25.md — Naming Patterns](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture-sephora-customer-segmentation-ml-2026-03-25.md — Error Handling Patterns](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6 (Bob / SM Agent — context preparation)
Claude Opus 4.6 (Amelia / Dev Agent — implementation)

### Debug Log References

- `subscription_date` column contains " UTC" suffix → `parse_dates` alone insufficient. Fixed with `pd.to_datetime(utc=True)` + `tz_localize(None)` post-processing.
- Columns 32-33 (`age_category`, `age_generation`) have mixed types → added explicit `dtype=str` overrides to suppress `DtypeWarning`.

### Completion Notes List

- Task 1: Created `src/` package (`__init__.py`, `config.py`, `data_loader.py`) and `data/processed/` directory.
- Task 2: `load_raw_data()` loads CSV with utf-8-sig encoding, string dtype overrides for ID columns, parse_dates for transactionDate/first_purchase_dt, and manual UTC parsing for subscription_date using `format="mixed"` to preserve fractional seconds.
- Task 3: `validate_schema()` checks column count (34), anonymized_card_code dtype (object), and datetime columns (including subscription_date). Raises ValueError with descriptive messages on failure. Prints summary on success.
- Task 4: Notebook updated with 3 cells — Setup (config + seed), Load & Validate (imports from src, displays shape/dtypes/head), Markdown observations (399,997 rows, 64,469 customers, 2025 date range).
- 15 unit tests created covering all ACs and edge cases — all pass quickly using a module-scoped pytest fixture.

### Senior Developer Review (AI)

**Review Date:** 2026-03-27
**Review Outcome:** Approved (Fixes Applied Automatically)
**Action Items:** 0 unchecked

#### Findings Fixed:
- 🔴 **HIGH:** `errors="coerce"` was silently dropping ~10,000 valid `subscription_date` entries with fractional seconds. Fixed by using `format="mixed"`.
- 🟡 **MEDIUM:** Test suite execution time was slow (~15s) due to calling `load_raw_data()` 13 times. Fixed by implementing a `@pytest.fixture(scope="module")`.
- 🟢 **LOW:** `validate_schema` did not verify `subscription_date` dtype, risking masked schema violations. Added validation check for it.

### File List

Files created:
- `src/__init__.py`
- `src/config.py`
- `src/data_loader.py`
- `data/processed/.gitkeep`
- `tests/test_data_loader.py`

Files modified:
- `first-analysis.ipynb` (replaced ad-hoc cell with structured E1-S1.1 section: Setup, Load & Validate, Markdown Observations)
