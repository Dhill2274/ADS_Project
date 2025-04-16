import pandas as pd
import numpy as np

# Load each correlation file
df_pearson = pd.read_csv("pearson_correlations.csv")
df_spearman = pd.read_csv("spearman_correlations.csv")
df_dcor = pd.read_csv("dcor_correlations.csv")

# Set threshold
threshold_min = 0.8
threshold_max = 1.0

# Filter: absolute values for pearson and spearman, raw for dcor (already positive)
pearson_filtered = df_pearson[(df_pearson['correlation'].abs() >= threshold_min) & (df_pearson['correlation'].abs() <= threshold_max)]
spearman_filtered = df_spearman[(df_spearman['correlation'].abs() >= threshold_min) & (df_spearman['correlation'].abs() <= threshold_max)]
dcor_filtered = df_dcor[(df_dcor['correlation'] >= threshold_min) & (df_dcor['correlation'] <= threshold_max)].copy()

# Create lookup dicts: (country, question, trade_type) -> correlation
pearson_dict = {(row['country'], row['question'], row['trade_type']): row['correlation'] for _, row in pearson_filtered.iterrows()}
spearman_dict = {(row['country'], row['question'], row['trade_type']): row['correlation'] for _, row in spearman_filtered.iterrows()}

# Add columns for match source and other correlation values
def enrich_row(row):
    key = (row['country'], row['question'], row['trade_type'])
    pearson_val = pearson_dict.get(key, np.nan)
    spearman_val = spearman_dict.get(key, np.nan)

    if not np.isnan(pearson_val) and not np.isnan(spearman_val):
        match_source = "Both"
    elif not np.isnan(pearson_val):
        match_source = "Pearson"
    elif not np.isnan(spearman_val):
        match_source = "Spearman"
    else:
        match_source = None

    return pd.Series({
        'Match_Source': match_source,
        'pearson_value': pearson_val,
        'spearman_value': spearman_val
    })

# Apply to dcor filtered rows
dcor_filtered[['Match_Source', 'pearson_value', 'spearman_value']] = dcor_filtered.apply(enrich_row, axis=1)

# Rename dcor correlation to dcor_value
dcor_filtered = dcor_filtered.rename(columns={'correlation': 'dcor_value'})

# Keep only matched rows
matching_rows = dcor_filtered[dcor_filtered['Match_Source'].notna()]

# Save to CSV
output_path = "matching_dcor_high_corr_with_source.csv"
matching_rows.to_csv(output_path, index=False)
print(f"Saved {len(matching_rows)} matching dcor rows to '{output_path}'")

