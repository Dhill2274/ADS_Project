import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sys, os

# Append parent directory for custom module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ess.dataset import Dataset

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

ESS_DATASETS = ["chars", "media", "politics", "socio", "values", "wellbeing"]

def map_round_to_year(round_num):
    mapping = {1: 2002, 2: 2004, 3: 2006, 4: 2008, 5: 2010,
               6: 2012, 7: 2014, 8: 2016, 9: 2018, 10: 2020, 11: 2022}
    return mapping.get(round_num, None)

def load_gdp_military_expenditure():
    # Load the wide-format GDP expenditure dataset
    df = pd.read_csv("Datasets/arms/gdp-expenditure.csv")
    # Melt the dataset so that each row is Country, Year, GDP_Expenditure
    df = df.melt(id_vars=["Country"], var_name="Year", value_name="GDP_Expenditure")
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df.dropna(subset=['Year'], inplace=True)
    df['Year'] = df['Year'].astype(int)
    # Replace "..." with NaN, remove trailing '%' and convert to float
    df['GDP_Expenditure'] = df['GDP_Expenditure'].replace('...', np.nan)
    df['GDP_Expenditure'] = df['GDP_Expenditure'].str.rstrip('%')
    df['GDP_Expenditure'] = pd.to_numeric(df['GDP_Expenditure'], errors='coerce')
    return df

def prepare_unified_dataset():
    """
    For this example, we use only the "values" ESS dataset and two human value questions.
    """
    gdp_exp_df = load_gdp_military_expenditure()
    unified_data = []

    # Use the "values" dataset from ESS
    dataset_obj = Dataset("values")
    questions = dataset_obj.questionLabels  # all questions in the "values" dataset
    answer_data = dataset_obj.questionsMean

    for country_code, country_name in ESS_COUNTRIES.items():
        if country_code not in dataset_obj.countryLabels:
            continue
        for round_num in range(1, 12):
            year = map_round_to_year(round_num)
            if year is None:
                continue

            # Fetch GDP expenditure for this country and year
            gdp_record = gdp_exp_df[(gdp_exp_df['Country'] == country_name) &
                                    (gdp_exp_df['Year'] == year)]
            if not gdp_record.empty:
                gdp_expenditure = gdp_record['GDP_Expenditure'].values[0]
            else:
                gdp_expenditure = np.nan

            data_row = {
                'Country': country_name,
                'Year': year,
                'GDP_Expenditure': gdp_expenditure,
                'ESS_Dataset': "values"
            }

            # We'll use two human values questions (adjust these codes if needed)
            for human_value_question in ['impenv', 'iplylfr']:
                # Get answers for this question for the country
                answers = answer_data[human_value_question].get(country_code, [])
                if len(answers) >= round_num:
                    value = answers[round_num - 1]
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        value = np.nan
                else:
                    value = np.nan
                data_row[human_value_question] = value

            unified_data.append(data_row)

    df_unified = pd.DataFrame(unified_data)
    # Drop rows with missing GDP expenditure so our model has a target
    df_unified.dropna(subset=['GDP_Expenditure'], inplace=True)
    df_unified.reset_index(drop=True, inplace=True)
    return df_unified

if __name__ == "__main__":
    # 1. Prepare unified dataset
    data = prepare_unified_dataset()
    
    # 2. Filter for Ireland and sort by Year
    ireland_df = data[data['Country'] == 'Ireland'].sort_values('Year').reset_index(drop=True)
    print("Ireland Data:")
    print(ireland_df[['Country', 'Year', 'GDP_Expenditure', 'impenv', 'iplylfr']])
    
    # 3. Prepare historical human values DataFrame for Ireland (using country code 'IE')
    years_hist = sorted(ireland_df['Year'].unique())
    human_values = ['impenv', 'iplylfr']
    human_values_hist = pd.DataFrame({'Year': years_hist})
    # Load human values from the "values" dataset for Ireland
    values_dataset = Dataset('values')
    answer_data = values_dataset.questionsMean
    for val in human_values:
        # Using country code 'IE' for Ireland
        ireland_answers = answer_data[val].get('IE', [np.nan]*len(years_hist))
        # Ensure the length is at least the number of historical years
        human_values_hist[val] = pd.to_numeric(ireland_answers[:len(years_hist)], errors='coerce')
    
    # Fill missing values (forward fill as a simple method)
    human_values_hist.fillna(method='ffill', inplace=True)
    
    print("\nHistorical Human Values:")
    print(human_values_hist)
    
    # 4. Forecast future human values using a simple linear regression model
    future_years = np.array([2024, 2026, 2028, 2030]).reshape(-1, 1)
    forecasted_human_values = {}
    for val in human_values:
        lr_model = LinearRegression()
        lr_model.fit(human_values_hist[['Year']], human_values_hist[val])
        forecasted_values = lr_model.predict(future_years)
        forecasted_human_values[val] = forecasted_values
    
    forecasted_human_df = pd.DataFrame({
        'Year': future_years.flatten(),
        'impenv': forecasted_human_values['impenv'],
        'iplylfr': forecasted_human_values['iplylfr']
    })
    
    print("\nForecasted Human Values for Future Years:")
    print(forecasted_human_df)
    
    # 5. Fit ARIMAX model on historical GDP with historical human values.
    # We'll use human values as exogenous variables.
    # For this ARIMAX, we'll only use the historical data.
    # (If you want to include the Year as a trend, you could add it as well.)
    endog = ireland_df['GDP_Expenditure']
    exog = human_values_hist[human_values]  # Historical human values
    
    # Check for missing values in exog and fill if necessary
    exog.fillna(exog.mean(), inplace=True)
    
    model = sm.tsa.statespace.SARIMAX(endog, exog=exog, order=(1, 1, 1))
    results = model.fit()
    print("\nARIMAX Model Summary:")
    print(results.summary())
    
    # 6. Forecast future GDP expenditure using the forecasted human values.
    # We need to supply exogenous data for future time periods.
    gdp_forecast = results.get_forecast(steps=len(future_years), exog=forecasted_human_df[human_values])
    forecast_mean = gdp_forecast.predicted_mean
    conf_int = gdp_forecast.conf_int()
    
    forecast_output_df = pd.DataFrame({
        'Year': future_years.flatten(),
        'Forecasted_GDP_Expenditure': forecast_mean.values,
        'Lower_CI': conf_int.iloc[:,0].values,
        'Upper_CI': conf_int.iloc[:,1].values
    })
    
    print("\nForecasted GDP Expenditure using Forecasted Human Values:")
    print(forecast_output_df)
