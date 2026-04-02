# Story 1.3: Customer-Level Aggregation

Status: done

## Story

As a Data Analyst,
I want to collapse transaction-level rows into one row per customer,
so that each observation in the feature matrix represents a unique individual.

## Acceptance Criteria

1. Output DataFrame has exactly one row per `anonymized_card_code`
2. Number of unique customers printed
3. All aggregations defined in the table below computed per customer:

| Output Column | Source | Aggregation |
|---|---|---|
| `total_transactions` | `anonymized_Ticket_ID` | `nunique` |
| `total_lines` | rows | `count` |
| `total_sales_eur` | `salesVatEUR` | `sum` |
| `avg_sales_eur` | `salesVatEUR` | `mean` |
| `total_discount_eur` | `discountEUR` | `sum` |
| `total_quantity` | `quantity` | `sum` |
| `last_purchase_date` | `transactionDate` | `max` |
| `first_purchase_date` | `transactionDate` | `min` |
| `loyalty_status` | `status` | `last` |
| `age` | `age` | `first` |
| `age_category` | `age_category` | `first` |
| `age_generation` | `age_generation` | `first` |
| `gender` | `gender` | `first` |
| `country` | `countryIsoCode` | `first` |
| `customer_city` | `customer_city` | `first` |
| `subscription_date` | `subscription_date` | `first` |
| `channel_recruitment` | `channel_recruitment` | `first` |
| `salesVatEUR_first_purchase` | `salesVatEUR_first_purchase` | `first` |
| `Axe_Desc_first_purchase` | `Axe_Desc_first_purchase` | `first` |

4. `anonymized_card_code` set as the DataFrame index
5. Aggregation logic documented in a Markdown cell

## Tasks / Subtasks

- [x] Task 1 — Implement `aggregate_to_customer_level()` in `src/feature_engineer.py` (AC: 1–4)
  - [x] Build aggregation dict with all columns from AC table
  - [x] Apply `groupby('anonymized_card_code').agg(agg_dict)`
  - [x] Rename columns to snake_case output names
  - [x] Set `anonymized_card_code` as index
  - [x] Print unique customer count
  - [x] Return `df_customers` (`pd.DataFrame`)
- [x] Task 2 — Add notebook section in `first-analysis.ipynb` (AC: 5)
  - [x] Call `aggregate_to_customer_level(df_clean)`
  - [x] Display `df_customers.shape`, `df_customers.head(3)`
  - [x] Markdown cell documenting aggregation logic

## Dev Notes

### Architecture Guardrails

**Module:** `src/feature_engineer.py` — add to existing module (created in US-1.2).  
**Notebook:** `01_eda.ipynb` — E1 section, after US-1.2 cells.

**Index convention (CRITICAL — from architecture doc):**
> Customer-level DataFrame index = `anonymized_card_code` (set as index, not column).
> All downstream modules expect this index convention.

**Aggregation dict pattern:**
```python
agg_dict = {
    'anonymized_Ticket_ID': ('total_transactions', 'nunique'),
    'anonymized_Ticket_ID': ('total_lines', 'count'),   # note: use size separately
    'salesVatEUR': [('total_sales_eur', 'sum'), ('avg_sales_eur', 'mean')],
    'discountEUR': ('total_discount_eur', 'sum'),
    'quantity': ('total_quantity', 'sum'),
    'transactionDate': [('last_purchase_date', 'max'), ('first_purchase_date', 'min')],
    'status': ('loyalty_status', 'last'),
    'age': ('age', 'first'),
    'age_category': ('age_category', 'first'),
    'age_generation': ('age_generation', 'first'),
    'gender': ('gender', 'first'),
    'countryIsoCode': ('country', 'first'),
    'customer_city': ('customer_city', 'first'),
    'subscription_date': ('subscription_date', 'first'),
    'channel_recruitment': ('channel_recruitment', 'first'),
    'salesVatEUR_first_purchase': ('salesVatEUR_first_purchase', 'first'),
    'Axe_Desc_first_purchase': ('Axe_Desc_first_purchase', 'first'),
}
```
Note: Use pandas NamedAgg syntax for clean multi-aggregation per column.

**total_lines** requires separate step (row count per customer):
```python
# total_lines = count of transaction lines per customer (computed before agg)
total_lines = df_clean.groupby('anonymized_card_code').size().rename('total_lines')
df_customers = df_customers.join(total_lines)
```

**is_click_collect aggregation** — also pass the C&C transaction count:
```python
cc_count = df_clean.groupby('anonymized_card_code')['is_click_collect'].sum().rename('cc_transactions')
df_customers = df_customers.join(cc_count)
```

### Previous Story Output (1-2)
- `df_clean`: cleaned transaction DataFrame with `is_click_collect` column and recoded `gender`/`status`
- `src/feature_engineer.py` already exists

### Output of This Story
- `df_customers`: customer-level aggregated DataFrame, index=`anonymized_card_code`
- Available for US-1.4 through US-1.7 (feature computation stories)

### Assertion Gate (from architecture patterns)
```python
assert df_customers.index.name == "anonymized_card_code"
assert df_customers.index.nunique() == len(df_customers)  # no duplicates
print(f"Unique customers: {len(df_customers):,}")
```

### References
- [epics — US-1.3](_bmad-output/planning-artifacts/epics-sephora-customer-segmentation-ml-2026-03-25.md)
- [architecture — DataFrame Format Patterns](_bmad-output/planning-artifacts/architecture-sephora-customer-segmentation-ml-2026-03-25.md)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6 (Implementation) / Gemini 3.1 Pro (Review Fixes)

### Debug Log References
None — clean implementation, no debugging needed.

### Senior Developer Review (AI)
- **Status:** Approved
- **Review Date:** 2026-03-27
- **Findings Fixed:** Edge case handling (NaNs in `.astype(int)`), missing edge case tests added, and hardcoded constants moved to config. Notebook cell updated to show all columns.

### Completion Notes List
- Task 1: Implemented `aggregate_to_customer_level()` using pandas NamedAgg syntax for all 18 AC columns. `total_lines` computed via `groupby.size()` (separate from agg). `cc_transactions` computed via `is_click_collect.sum()`. Assertion gate validates index name and uniqueness. 18 unit tests cover every aggregation column, index convention, and edge cases. 53/53 tests pass (0 regressions).
- Task 2: Added Cell 5 (code) and Markdown cell documenting all aggregation logic in `first-analysis.ipynb`, after US-1.2 section.

### File List
Files modified:
- `src/feature_engineer.py` — added `aggregate_to_customer_level()` function
- `tests/test_feature_engineer.py` — added `TestAggregateToCustomerLevel` class (18 tests)
- `first-analysis.ipynb` — added US-1.3 aggregation code cell + documentation markdown cell
