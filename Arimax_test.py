import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import sys, os
import matplotlib.pyplot as plt

# Add src to path for Dataset import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from ess.dataset import Dataset

# Constants
ESS_COUNTRIES = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CH": "Switzerland",
    "CY": "Cyprus", "CZ": "Czechia", "DE": "Germany", "DK": "Denmark",
    "EE": "Estonia", "ES": "Spain", "FI": "Finland", "FR": "France",
    "GB": "United Kingdom", "HU": "Hungary", "IE": "Ireland", "IL": "Israel",
    "LT": "Lithuania", "NL": "Netherlands", "NO": "Norway", "PL": "Poland",
    "PT": "Portugal", "RU": "Russian Federation", "SE": "Sweden",
    "SI": "Slovenia", "SK": "Slovakia", "UA": "Ukraine"
}

ESS_DATASETS = ["politics", "wellbeing", "chars"]

def map_round_to_year(round_num):
    mapping = {1: 2002, 2: 2004, 3: 2006, 4: 2008, 5: 2010,
               6: 2012, 7: 2014, 8: 2016, 9: 2018, 10: 2020, 11: 2022}
    return mapping.get(round_num, None)

def load_gdp_military_expenditure():
    # Use the correct Excel file and read with pandas
    gdp_path = "/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Desktop/University Engmaths/lets get it to work/2002-2023 final set.xlsx"
    df = pd.read_excel(gdp_path)
    # Melt the dataset so that each row is Country, Year, GDP_Expenditure
    df = df.melt(id_vars=["Country"], var_name="Year", value_name="GDP_Expenditure")
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df.dropna(subset=['Year'], inplace=True)
    df['Year'] = df['Year'].astype(int)
    df['GDP_Expenditure'] = pd.to_numeric(df['GDP_Expenditure'], errors='coerce')
    return df

def prepare_unified_dataset_with_topic_means():
    gdp_exp_df = load_gdp_military_expenditure()
    unified_data = []
    # Prepare Dataset objects for each topic
    dataset_objs = {ds: Dataset(ds) for ds in ESS_DATASETS}

    for country_name in gdp_exp_df['Country'].unique():
        # Map country name to code
        country_code = None
        for code, name in ESS_COUNTRIES.items():
            if name == country_name:
                country_code = code
                break
        if not country_code:
            continue

        gdp_country = gdp_exp_df[gdp_exp_df['Country'] == country_name]
        years = sorted(gdp_country['Year'].unique())
        exog_dict = {}

        # For each topic, get the mean time series for this country
        for ds in ESS_DATASETS:
            topic_means = dataset_objs[ds].get_topic_mean_timeseries(country_code)
            # Align with years
            exog_dict[ds] = [topic_means.get(y, np.nan) for y in years]

        for i, year in enumerate(years):
            gdp_row = gdp_country[gdp_country['Year'] == year]
            if gdp_row.empty:
                continue
            gdp_expenditure = gdp_row['GDP_Expenditure'].values[0]
            row_data = {
                'Country': country_name,
                'Country_Code': country_code,
                'Year': year,
                'GDP_Expenditure': gdp_expenditure
            }
            # Add exogenous variables for each dataset
            for ds in ESS_DATASETS:
                row_data[f"{ds}_mean"] = exog_dict[ds][i] if i < len(exog_dict[ds]) else np.nan
            unified_data.append(row_data)
    df_unified = pd.DataFrame(unified_data)
    df_unified.dropna(subset=['GDP_Expenditure'], inplace=True)
    df_unified.reset_index(drop=True, inplace=True)
    return df_unified

if __name__ == "__main__":
    # 1. Prepare unified dataset using mean topic time series for each country
    data = prepare_unified_dataset_with_topic_means()

    # 2. For each country, fit ARIMAX and forecast
    forecast_years = [2024, 2026, 2028, 2030]
    results_all = []

    for country in data['Country'].unique():
        df_country = data[data['Country'] == country].sort_values('Year').reset_index(drop=True)
        exog_cols = [f"{ds}_mean" for ds in ESS_DATASETS]
        if df_country[exog_cols].isnull().all(axis=None):
            continue  # Skip if no exogenous data

        # Prepare exogenous variables (mean topic time series)
        exog = df_country[exog_cols].copy()
        exog.ffill(inplace=True)
        exog.fillna(exog.mean(), inplace=True)

        # Prepare endogenous variable (GDP_Expenditure)
        endog = df_country['GDP_Expenditure']

        # Forecast future exogenous values using linear regression for each variable
        ess_future = []
        for col in exog_cols:
            exog_col = exog[col].fillna(0)
            if exog_col.isnull().all():
                ess_future.append([0] * len(forecast_years))
                continue
            lr_model = LinearRegression()
            lr_model.fit(df_country[['Year']], exog_col)
            future_years_df = pd.DataFrame({'Year': forecast_years})
            ess_future.append(lr_model.predict(future_years_df))
        ess_future = np.column_stack(ess_future)

        # Fit ARIMAX
        try:
            model = sm.tsa.statespace.SARIMAX(endog, exog=exog, order=(1, 1, 1))
            results = model.fit(disp=False)
        except Exception as e:
            print(f"ARIMAX failed for {country}: {e}")
            continue

        # Forecast future GDP expenditure
        exog_future_df = pd.DataFrame(ess_future, columns=exog_cols)
        gdp_forecast = results.get_forecast(steps=len(forecast_years), exog=exog_future_df)
        forecast_mean = gdp_forecast.predicted_mean
        conf_int = gdp_forecast.conf_int()

        for i, year in enumerate(forecast_years):
            results_all.append({
                'Country': country,
                'Year': year,
                'Forecasted_GDP_Expenditure': forecast_mean.iloc[i],
                'Lower_CI': conf_int.iloc[i, 0],
                'Upper_CI': conf_int.iloc[i, 1]
            })

    forecast_output_df = pd.DataFrame(results_all)
    print("\nForecasted GDP Expenditure for All Countries:")
    print(forecast_output_df)

    # Plot forecast for selected countries only
    plot_countries = ["Ukraine", "United Kingdom"]  # Order matters: Ukraine above UK

    # Prepare data for subplots
    subplot_data = []
    for country in plot_countries:
        df_hist = data[data['Country'] == country].sort_values('Year')
        df_fore = forecast_output_df[forecast_output_df['Country'] == country].sort_values('Year')
        if df_hist.empty or df_fore.empty:
            continue
        subplot_data.append((country, df_hist, df_fore))

    if subplot_data:
        fig, axes = plt.subplots(nrows=len(subplot_data), ncols=1, figsize=(8, 10), sharex=False)
        if len(subplot_data) == 1:
            axes = [axes]
        for ax, (country, df_hist, df_fore) in zip(axes, subplot_data):
            # Print exogenous variable (topic) time series used for this country
            print(f"\nExogenous variables (topics) for {country}:")
            for ds in ESS_DATASETS:
                topic_series = df_hist[f"{ds}_mean"].values
                print(f"  {ds}: {topic_series}")

            years_all = pd.concat([df_hist['Year'], df_fore['Year']])
            values_all = pd.concat([df_hist['GDP_Expenditure'], df_fore['Forecasted_GDP_Expenditure']])
            # Plot historical (red) and forecast (blue) separately
            ax.plot(df_hist['Year'], df_hist['GDP_Expenditure'], color='red', linewidth=2, label='Historical')
            ax.plot(df_fore['Year'], df_fore['Forecasted_GDP_Expenditure'], color='blue', linewidth=2, label='Forecast')
            ax.scatter(df_hist['Year'], df_hist['GDP_Expenditure'], marker='o', color='red', s=60)
            ax.scatter(df_fore['Year'], df_fore['Forecasted_GDP_Expenditure'], marker='D', color='blue', s=70)
            # Title with line break
            ax.set_title(f"{country} Military Expenditure\n(% GDP) Forecast via ARIMAX", fontsize=20, fontweight='bold')
            ax.set_ylabel("Military Expenditure (% GDP)", fontsize=18)
            ax.set_xlabel("Year", fontsize=18)
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(fontsize=16, loc='best', frameon=True)
        plt.tight_layout()
        plt.savefig("Ukraine_and_United_Kingdom_forecast_arimax.png", dpi=150)
        plt.show()
