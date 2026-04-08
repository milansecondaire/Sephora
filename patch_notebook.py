import json

with open('02_clustering.ipynb', 'r') as f:
    nb = json.load(f)

# Find the cell doing to_csv
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        modified = False
        for line in cell['source']:
            if 'df_customers.to_csv' in line and 'customers_with_clusters.csv' in line:
                # Insert index handling before to_csv
                new_source.append('if df_customers.index.name == "anonymized_card_code":\n')
                new_source.append('    df_customers = df_customers.reset_index()\n')
                new_source.append('elif "anonymized_card_code" not in df_customers.columns:\n')
                new_source.append('    print("⚠️ WARNING: anonymized_card_code not found in index or columns. PK lost.")\n')
                new_source.append(line)
                modified = True
            else:
                new_source.append(line)
        if modified:
            cell['source'] = new_source

with open('02_clustering.ipynb', 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

