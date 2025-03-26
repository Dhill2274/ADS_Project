import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys, os

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
    # Remove trailing '%' and convert to float; treat '...' as missing
    df['GDP_Expenditure'] = df['GDP_Expenditure'].replace('...', np.nan)
    df['GDP_Expenditure'] = df['GDP_Expenditure'].str.rstrip('%')
    df['GDP_Expenditure'] = pd.to_numeric(df['GDP_Expenditure'], errors='coerce')
    return df

def prepare_unified_dataset():
    gdp_exp_df = load_gdp_military_expenditure()
    unified_data = []

    # For this example, we'll focus on the "values" ESS dataset.
    # (You can extend this to merge across datasets if desired.)
    dataset_obj = Dataset("values")
    questions = dataset_obj.questionLabels  # all questions from the "values" dataset
    answer_data = dataset_obj.questionsMean

    for country_code, country_name in ESS_COUNTRIES.items():
        if country_code not in dataset_obj.countryLabels:
            continue
        for round_num in range(1, 12):
            year = map_round_to_year(round_num)
            if year is None:
                continue

            # Get GDP expenditure for this country and year
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

            # Add selected human values questions (e.g., "impenv" and "iplylfr")
            for human_value_question in ['impenv', 'iplylfr']:
                # Get answers for this question for the country
                # Note: answer_data is a dictionary-like object keyed by question code.
                answers = answer_data[human_value_question].get(country_code, [])
                if len(answers) >= round_num:
                    value = answers[round_num - 1]
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        value = np.nan
                else:
                    value = np.nan
                # Use the human value question code as the key
                data_row[human_value_question] = value

            unified_data.append(data_row)

    df_unified = pd.DataFrame(unified_data)
    # Drop rows with missing GDP expenditure
    df_unified.dropna(subset=['GDP_Expenditure'], inplace=True)
    df_unified.reset_index(drop=True, inplace=True)
    return df_unified

if __name__ == "__main__":
    data = prepare_unified_dataset()
    
    # Filter for Ireland and sort by Year
    ireland_df = data[data['Country'] == 'Ireland'].sort_values('Year')
    # For example, fill missing values with the mean of the column
    ireland_df[['impenv', 'iplylfr']] = ireland_df[['impenv', 'iplylfr']].fillna(ireland_df[['impenv', 'iplylfr']].mean())

    print("Ireland Data:")
    print(ireland_df[['Country', 'Year', 'GDP_Expenditure', 'impenv', 'iplylfr']])
    
    # Define dependent (endogenous) and independent (exogenous) variables.
    # Here we use GDP_Expenditure as the dependent variable and the two human value questions as exog.
    endog = ireland_df['GDP_Expenditure']
    # Use the actual column names as stored in your unified dataset ("impenv" and "iplylfr")
    exog = ireland_df[['impenv', 'iplylfr']]

    # Fit ARIMAX model (you may need to adjust order based on your data characteristics)
    model = sm.tsa.statespace.SARIMAX(endog, exog=exog, order=(1, 1, 1))
    results = model.fit()

    # Print model summary
    print(results.summary())

