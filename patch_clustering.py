import re

with open("src/clustering.py", "r") as f:
    code = f.read()

# 1. Add "Notes" column to build_comparison_table
old_return_comp_df = 'comp_df["min_cluster_pct"] = min_pcts\n    return comp_df'
new_return_comp_df = 'comp_df["min_cluster_pct"] = min_pcts\n    if "notes" not in comp_df.columns:\n        comp_df["notes"] = ""\n    return comp_df'
code = code.replace(old_return_comp_df, new_return_comp_df)

# 2. Fix select_best_algorithm: include CH score and handle DB max=0
old_score = '    df["score"] = df["silhouette"] - df["davies_bouldin"] / db_max'
new_score = '''    # Score incorporates silhouette, normalized negative DB, and normalized CH
    ch_max = df["calinski_harabasz"].max()
    db_norm = df["davies_bouldin"] / db_max if db_max > 0 else 0
    ch_norm = df["calinski_harabasz"] / ch_max if ch_max > 0 else 0
    df["score"] = df["silhouette"] - db_norm + 0.1 * ch_norm
'''
code = code.replace(old_score, new_score)

with open("src/clustering.py", "w") as f:
    f.write(code)

