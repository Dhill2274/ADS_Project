import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Path to the Excel file
file_path = "Data/2002-2023 expenditure gdp % europe .xlsx"

# Read the data
df = pd.read_excel(file_path)

# Extract year columns (assumes first column is 'Country')
years = [int(col) for col in df.columns[1:]]

# Prepare output dictionary
extrapolated = {}

future_years = np.arange(years[-1] + 1, 2031)  # 2024-2030

plot_countries = ["Ukraine", "United Kingdom"]

fig, axes = plt.subplots(2, 1, figsize=(8, 10))  # Remove sharey=True

for ax_idx, country in enumerate(plot_countries):
    row = df[df['Country'] == country].iloc[0]
    y = row.iloc[1:].values.astype(float)
    X = np.array(years).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    # Extrapolate for 2025
    y_pred = model.predict(np.array([[2025]]))[0]
    extrapolated[country] = y_pred
    # Predict for all years including future
    all_years = np.concatenate([years, future_years])
    y_all = model.predict(all_years.reshape(-1, 1))
    
    ax = axes[ax_idx]
    
    # Full regression line in black
    ax.plot(all_years, y_all, color='black', linewidth=2, label='Regression Line')
    
    # Historical data - scatter only
    ax.scatter(years, y, marker='o', color='red', s=60, label='Historical')
    
    # Forecast data - scatter only
    ax.scatter(future_years, y_forecast, marker='D', color='blue', s=70, label='Forecast')
    
    # Title with specified format
    ax.set_title(f"{country} Military Expenditure\n(% GDP) Forecast via Linear Regression", 
                 fontsize=20, fontweight='bold')
    
    # Axis labels
    ax.set_xlabel("Year", fontsize=18)
    ax.set_ylabel("Military Expenditure (% GDP)", fontsize=18)
    
    # Ticks
    ax.set_xticks(np.arange(2000, 2031, 5))
    ax.set_xlim(min(years), 2030)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Legend
    ax.legend(fontsize=16, loc='best', frameon=True)
    
    # Custom y-limits for the UK plot
    if country == "United Kingdom":
        min_val = min(np.min(y), np.min(y_all))
        max_val = max(np.max(y), np.max(y_all))
        y_range = max_val - min_val
        padding = y_range * 0.2
        ax.set_ylim(min_val - padding, max_val + padding)

plt.tight_layout()
plt.savefig("Combined_forecast_linearregression.png")
plt.show()

# Print results
print("Extrapolated expenditure (% GDP) for 2025:")
for country, value in extrapolated.items():
    print(f"{country}: {value:.2f}")
