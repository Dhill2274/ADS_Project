import pandas as pd
import numpy as np

# Read in the two datasets
df_house = pd.read_csv("Combined/combined_ess(household).csv", sep='\t')
df_media = pd.read_csv("Cleaned/Cleaned_Media&Trust.csv", sep=',')

df_house.columns = df_house.columns.str.replace(r"^'|'$", "", regex=True)

# print(df_house.columns.tolist())
# print(df_media.columns.tolist())
# print([repr(col) for col in df_house.columns])
# print([repr(col) for col in df_media.columns])

# Keep only the relevant columns from the media dataset
df_media = df_media[['name', 'essround', 'cntry', 'pplfair', 'pplhlp', 'ppltrst']]

# Group both dataframes by the matching keys
group_house = df_house.groupby(['name', 'essround', 'cntry'])
group_media = df_media.groupby(['name', 'essround', 'cntry'])

# Get the set of all keys from both datasets
keys_house = set(group_house.groups.keys())
keys_media = set(group_media.groups.keys())
all_keys = keys_house.union(keys_media)

# List to collect combined rows
combined_rows = []

# Loop over each key (which is a tuple: (name, essround, cntry))
for key in all_keys:
    # For each key, get the corresponding groups (if present)
    house_group = group_house.get_group(key) if key in group_house.groups else None
    media_group = group_media.get_group(key) if key in group_media.groups else None

    n_house = len(house_group) if house_group is not None else 0
    n_media = len(media_group) if media_group is not None else 0

    # The number of rows for this key in the combined dataset is the maximum of the two counts.
    n_rows = max(n_house, n_media)
    
    for i in range(n_rows):
        # Build a new row dictionary for the combined dataset
        row = {
            "name": key[0],
            "essround": key[1],
            "cntry": key[2]
        }
        
        # Add household independent variables (hhmmb and agea) if available.
        # Only fill them for the first row from the household group; if there are extra media rows, leave these as None.
        if house_group is not None and i < n_house:
            row["hhmmb"] = house_group.iloc[i]["hhmmb"]
            row["agea"] = house_group.iloc[i]["agea"]
        else:
            row["hhmmb"] = None
            row["agea"] = None

        # Add media & trust independent variables (pplfair, pplhlp, ppltrst) if available.
        if media_group is not None and i < n_media:
            row["pplfair"] = media_group.iloc[i]["pplfair"]
            row["pplhlp"] = media_group.iloc[i]["pplhlp"]
            row["ppltrst"] = media_group.iloc[i]["ppltrst"]
        else:
            row["pplfair"] = None
            row["pplhlp"] = None
            row["ppltrst"] = None

        combined_rows.append(row)

# Create the combined dataframe with the desired columns
combined_df = pd.DataFrame(combined_rows, 
                           columns=["name", "essround", "cntry", "hhmmb", "agea", "pplfair", "pplhlp", "ppltrst"])

# List of columns to process
cols_to_update = ['pplfair', 'pplhlp', 'ppltrst']

# Replace 77, 88, and 99 with np.nan in the specified columns
combined_df[cols_to_update] = combined_df[cols_to_update].replace([77, 88, 99], np.nan)

# Save the combined dataframe to a new CSV file
combined_df.to_csv("Combined/combined_ess(houshold_media).csv", index=False)

print("Combined CSV created successfully as 'combined_ess(houshold_media).csv'")
