import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
from matplotlib.colors import PowerNorm  # <-- add this import

# --- 1. Load Data ---
# Adjust the filename to the correct path
file_path = '/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Desktop/University Engmaths/Applied data science/ADS Coursework/Data/All_countries_headers_removed_imputed.csv'
try:
    # read_csv should handle empty fields as NaN by default
    df = pd.read_csv(file_path, index_col='Country')
except FileNotFoundError:
    print(f"Error: '{file_path}' not found.")
    print("Please make sure the data file exists at the specified path.")
    # Exit or use dummy data if needed
    exit()

# --- 2. Clean Data ---
# Ensure all year columns are numeric, coercing errors to NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Ensure column names (years) are integers
df.columns = df.columns.astype(int)

# --- 2.5 Filter Countries ---
# Define the list of countries to display.
# Modify this list to include/exclude countries as needed.
# To exclude Kuwait later, simply remove 'Kuwait' from this list.
countries_to_show = [
    'Kuwait', 'Ukraine', 'Eritrea', 'United States', 'Russia',
    'Saudi Arabia', 'Iran',
    'Israel' # Add or remove countries here
]

# Filter the DataFrame to keep only the selected countries
# Use .loc and check if the index (Country) is in the list
# Also check if the country actually exists in the DataFrame to avoid errors
available_countries_to_show = [c for c in countries_to_show if c in df.index]
if not available_countries_to_show:
    print("Error: None of the specified countries_to_show were found in the data.")
    exit()
df_filtered = df.loc[available_countries_to_show]

# --- 3. Prepare for Plotting ---
# Use the filtered DataFrame (df_filtered) instead of the original df
years = df_filtered.columns.values
countries = df_filtered.index.values # Now uses filtered list
country_indices = np.arange(len(countries)) # Use numerical indices for Y axis

# Create meshgrid
X, Y = np.meshgrid(years, country_indices)

# Get expenditure values (Z) from the filtered DataFrame
Z = df_filtered.values

# Fill NaN values with 0 for continuous surface plotting
Z_filled = np.nan_to_num(Z, nan=0.0)

# Compute min/max for color scaling (ignore zeros and NaNs)
Z_nonzero = Z[np.isfinite(Z) & (Z > 0)]
vmin = Z_nonzero.min() if Z_nonzero.size > 0 else 1e-2
vmax = Z_nonzero.max() if Z_nonzero.size > 0 else 1

# --- 4. Create 3D Plot ---
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111, projection='3d')

# Define a list of distinct colors (one per country)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink']
# Extend or modify the color list as needed for more countries

# Plot each country's data as a 3D line
for idx, country in enumerate(countries):
    ax.plot(
        years,                          # X: years
        np.full_like(years, idx),       # Y: country index (constant)
        Z_filled[idx, :],               # Z: values for this country
        color=colors[idx % len(colors)],
        label=country,
        linewidth=3
    )

# Label Y axis with country names
ax.set_yticks(country_indices)
ax.set_yticklabels(countries)

# Optionally, add a legend
from matplotlib.patches import Patch
legend_handles = [Patch(color=colors[i % len(colors)], label=countries[i]) for i in range(len(countries))]
# Place legend below the plot, centered, in a single row
ax.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=len(countries),
    frameon=False
)

ax.set_xlabel('Year')
ax.set_zlabel('Percent of GDP Towards Military Expenditure')

plt.show()
