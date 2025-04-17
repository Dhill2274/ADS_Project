import pandas as pd
import numpy as np
import sys
import os
from scipy import stats
import dcor
from collections import defaultdict

# Add the parent directory (workspace root) to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

# Import from ESS module
try:
    from src.ess.dataset import Dataset, DATASET_DISPLAY_NAMES
except ImportError:
    print("Error: Could not import Dataset module. Make sure 'src' is in the Python path.")
    print("Current sys.path:", sys.path)
    sys.exit(1)

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
NAME_TO_CODE = {name: code for code, name in ESS_COUNTRIES.items()}
MIN_DATA_POINTS = 3

def map_round_to_year(round_num):
    return {
        1: 2002, 2: 2004, 3: 2006, 4: 2008, 5: 2010,
        6: 2012, 7: 2014, 8: 2016, 9: 2018, 10: 2020, 11: 2022
    }.get(round_num)

def load_gdp_data(path="cleaned/arms/2002-2023_expenditure_gdp_%_europe.xlsx"):
    print(f"Loading GDP military expenditure data from {path}...")
    try:
        df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
        df = df.set_index("Country")
        df.columns = df.columns.astype(str)
        df = df.apply(pd.to_numeric, errors="coerce")
        print(f"Loaded data for {len(df)} countries and {len(df.columns)} years.")
        return df
    except Exception as e:
        print(f"Error loading GDP data: {e}")
        sys.exit(1)

def process_gdp_data(df):
    print("Processing GDP data into timeseries format...")
    gdp_dict = {}
    all_years = sorted([int(col) for col in df.columns if col.isdigit()])
    for country_name in df.index:
        if country_name in NAME_TO_CODE:
            year_dict = {}
            for year in all_years:
                value = df.loc[country_name, str(year)]
                if pd.notna(value):
                    year_dict[year] = value
            gdp_dict[country_name] = year_dict
    print(f"Processed GDP data for {len(gdp_dict)} ESS countries.")
    return all_years, gdp_dict

def calculate_all_correlations():
    gdp_df = load_gdp_data()
    years, gdp_data = process_gdp_data(gdp_df)
    min_year = min(years)
    max_year = max(years)

    pearson_correlations = []
    spearman_correlations = []
    dcor_correlations = []

    for dataset_name in ESS_DATASETS:
        print(f"Processing Dataset: {dataset_name}")
        try:
            dataset_obj = Dataset(dataset_name)
        except Exception as e:
            print(f"  Skipping dataset {dataset_name} due to load error: {e}")
            continue

        questions = dataset_obj.questionLabels
        answer_source = dataset_obj.questionsMean

        for country_code, country_name in ESS_COUNTRIES.items():
            if country_code not in dataset_obj.countryLabels or country_name not in gdp_data:
                continue

            for question in questions:
                if not isinstance(question, str):
                    continue

                original_col = dataset_obj._get_original_column(question)
                if original_col not in dataset_obj.df.columns:
                    continue

                try:
                    question_data_all_countries = answer_source[question]
                except KeyError:
                    continue

                question_timeseries = []

                for round_num in range(1, 12):
                    year = map_round_to_year(round_num)
                    if year is None or not (min_year <= year <= max_year):
                        continue

                    gdp_value = gdp_data.get(country_name, {}).get(year, 0)

                    ess_value = None
                    if country_code in question_data_all_countries:
                        try:
                            answers = question_data_all_countries[country_code]
                            if round_num - 1 < len(answers):
                                raw_val = answers[round_num - 1]
                                if raw_val != "None" and raw_val is not None:
                                    ess_value = float(raw_val)
                        except Exception:
                            continue

                    if ess_value is not None:
                        question_timeseries.append({'Year': year, 'GDP': gdp_value, 'ESS': ess_value})

                if len(question_timeseries) >= MIN_DATA_POINTS:
                    df_q = pd.DataFrame(question_timeseries)
                    if df_q['GDP'].nunique() <= 1 or df_q['ESS'].nunique() <= 1:
                        continue

                    try:
                        pearson_corr = df_q['GDP'].corr(df_q['ESS'], method='pearson')
                        spearman_corr, _ = stats.spearmanr(df_q['GDP'], df_q['ESS'])
                        dist_corr = dcor.distance_correlation(df_q['GDP'].to_numpy(), df_q['ESS'].to_numpy())

                        if pd.notna(pearson_corr) and np.isfinite(pearson_corr):
                            pearson_correlations.append({
                                'country': country_name,
                                'question': question,
                                'correlation': pearson_corr,
                                'data_points': len(df_q),
                                'dataset': dataset_name,
                                'trade_type': 'gdp',
                                'answer_type': 'mean'
                            })

                        if pd.notna(spearman_corr) and np.isfinite(spearman_corr):
                            spearman_correlations.append({
                                'country': country_name,
                                'question': question,
                                'correlation': spearman_corr,
                                'data_points': len(df_q),
                                'dataset': dataset_name,
                                'trade_type': 'gdp',
                                'answer_type': 'mean'
                            })

                        if pd.notna(dist_corr) and np.isfinite(dist_corr):
                            dcor_correlations.append({
                                'country': country_name,
                                'question': question,
                                'correlation': dist_corr,
                                'data_points': len(df_q),
                                'dataset': dataset_name,
                                'trade_type': 'gdp',
                                'answer_type': 'mean'
                            })

                    except Exception:
                        continue

    return pearson_correlations, spearman_correlations, dcor_correlations

def main(output_prefix="correlations_gdp"):
    print("Starting correlation analysis...")
    pearson_results, spearman_results, dcor_results = calculate_all_correlations()

    def save_results(data, filename_prefix, sort_by_abs=False):
        if not data:
            print(f"No valid {filename_prefix} correlations found.")
            return
        df = pd.DataFrame(data)
        if sort_by_abs:
            df['abs_correlation'] = df['correlation'].abs()
            df = df.sort_values(by='abs_correlation', ascending=False).drop(columns=['abs_correlation'])
        else:
            df = df.sort_values(by='correlation', ascending=False)

        filename = f"{filename_prefix}_{output_prefix}.csv"
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', filename)
        try:
            df.to_csv(output_path, index=False, float_format='%.5f')
            print(f"Saved {len(df)} {filename_prefix} correlations to {output_path}")
        except Exception as e:
            print(f"Error saving {filename_prefix} correlations: {e}")

    save_results(pearson_results, "pearson_gdp", sort_by_abs=True)
    save_results(spearman_results, "spearman_gdp", sort_by_abs=True)
    save_results(dcor_results, "dcor_gdp", sort_by_abs=False)

if __name__ == "__main__":
    main()
