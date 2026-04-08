with open('_bmad-output/implementation-artifacts/4-5-algorithm-comparison-final-selection.md', 'r') as f:
    content = f.read()

# Update status
content = content.replace('Status: in-progress', 'Status: done')
content = content.replace('Status: review', 'Status: done')

# Add AI Review section
action_items = """### Review Notes
- **BUG FIX (review):** `cluster_id` was set to string `f"C{x}"` instead of int alias. Fixed: `df_customers["cluster_id"] = df_customers["final_cluster"]` — matches architecture spec (int, 0-indexed).
- **BUG FIX (review)**: `Notes` column added to `build_comparison_table()` to satisfy AC-1.
- **BUG FIX (review)**: Calinski-Harabasz score factored into `select_best_algorithm()` to use all metrics. Division by zero on DB max guarded.
- **BUG FIX (review)**: `02_clustering.ipynb` export logic protected against dropping PK index (`anonymized_card_code`) unintentionally with a reset check.
"""
content = content.replace('### Review Notes\n- **BUG', action_items.replace('### Review Notes\n', '### Review Notes\n- **BUG', 1))

with open('_bmad-output/implementation-artifacts/4-5-algorithm-comparison-final-selection.md', 'w') as f:
    f.write(content)

