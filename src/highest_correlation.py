import pandas as pd
import numpy as np
import json
import sys
import os
import math
from collections import defaultdict

# Add the parent directory (workspace root) to the path
# Assumes this script is run from the 'src' directory
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
CODE_TO_NAME = {code: name for code, name in ESS_COUNTRIES.items()}
NAME_TO_CODE = {name: code for code, name in ESS_COUNTRIES.items()}
MIN_DATA_POINTS = 3 # Minimum number of overlapping years required for correlation


def load_trade_data(path="cleaned/arms/trades-clean.csv"):
    """Loads and cleans the trade data."""
    print(f"Loading trade data from {path}...")
    try:
        df = pd.read_csv(path)
        df = df[df['Quantity'] > 0]
        df['Year of order'] = pd.to_numeric(df['Year of order'], errors='coerce')
        df = df.dropna(subset=['Year of order'])
        df['Year of order'] = df['Year of order'].astype(int)
        print(f"Loaded {len(df)} trade records.")
        return df
    except FileNotFoundError:
        print(f"Error: Trade data file not found at {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading trade data: {e}")
        sys.exit(1)

def process_quantity_data(df):
    """Processes trade data into yearly quantities per country."""
    print("Processing quantity data...")
    years = sorted(df['Year of order'].unique())
    exporters = df['From'].unique()
    importers = df['To'].unique()
    all_countries = sorted(set(list(exporters) + list(importers)))

    sent_per_country = defaultdict(lambda: {year: 0 for year in years})
    received_per_country = defaultdict(lambda: {year: 0 for year in years})

    for _, row in df.iterrows():
        year = int(row['Year of order'])
        from_country = row['From']
        to_country = row['To']
        quantity = row['Quantity']
        if from_country in NAME_TO_CODE: # Only process countries in ESS_COUNTRIES
            sent_per_country[from_country][year] += quantity
        if to_country in NAME_TO_CODE: # Only process countries in ESS_COUNTRIES
            received_per_country[to_country][year] += quantity

    print(f"Processed data for {len(all_countries)} countries across {len(years)} years.")
    return years, all_countries, dict(sent_per_country), dict(received_per_country)

def map_round_to_year(round_num):
    """Maps ESS survey rounds to approximate years."""
    mapping = {
        1: 2002, 2: 2004, 3: 2006, 4: 2008, 5: 2010,
        6: 2012, 7: 2014, 8: 2016, 9: 2018, 10: 2020, 11: 2022
    }
    return mapping.get(round_num)

def calculate_all_correlations():
    """Calculates correlations for all datasets, trade types, and answer types."""
    trade_df = load_trade_data()
    years, _, sent_data, received_data = process_quantity_data(trade_df)
    min_year = min(years)
    max_year = max(years)

    all_correlations = []
    processed_count = 0

    for dataset_name in ESS_DATASETS:
        print(f"Processing Dataset: {dataset_name}")
        try:
            dataset_obj = Dataset(dataset_name)
        except Exception as e:
            print(f"  Skipping dataset {dataset_name} due to load error: {e}")
            continue

        questions = dataset_obj.questionLabels # Use all questions

        for trade_type in ['sent', 'received']:
            print(f"  Processing Trade Type: {trade_type}")
            trade_data = sent_data if trade_type == 'sent' else received_data

            # Only process the 'mean' answer type
            for answer_type in ['mean']:
                print(f"    Processing Answer Type: {answer_type}")
                answer_source = dataset_obj.questionsMean if answer_type == 'mean' else dataset_obj.questions

                for country_code, country_name in ESS_COUNTRIES.items():
                    if country_code not in dataset_obj.countryLabels or country_name not in trade_data:
                        continue

                    # Calculate correlation per question
                    for question in questions:
                        # Ensure question is a string before proceeding
                        if not isinstance(question, str):
                            # print(f"  Skipping non-string question label: {question} ({type(question)}) in dataset {dataset_name}")
                            continue

                        # Get the original column name for this question label
                        original_col = dataset_obj._get_original_column(question)

                        # Ensure the original column actually exists in the DataFrame
                        if original_col not in dataset_obj.df.columns:
                            # print(f"  Skipping question '{question}' (original: '{original_col}') as it's not in the DataFrame columns for dataset {dataset_name}")
                            continue

                        # Rebuild specific timeseries for this question
                        question_timeseries = []

                        # Fetch the data for this specific question once (now safe due to checks above)
                        try:
                            question_data_all_countries = answer_source[question]
                        except KeyError:
                             # This should ideally not happen due to the checks, but safeguard
                            # print(f"  Skipping question '{question}' due to KeyError during data fetch.")
                            continue

                        for round_num in range(1, 12):
                            year = map_round_to_year(round_num)
                            if year is None or not (min_year <= year <= max_year):
                                continue

                            trade_quantity = trade_data.get(country_name, {}).get(year, 0)

                            ess_value = None
                            # Access the pre-fetched data for the current country and round
                            if country_code in question_data_all_countries:
                                try:
                                    # Get the list of answers for this country for the current question
                                    answer_list = question_data_all_countries[country_code]
                                    # Check if data for the specific round exists
                                    if round_num -1 < len(answer_list):
                                        raw_val = answer_list[round_num - 1]
                                        if raw_val != "None" and raw_val is not None:
                                            try:
                                                ess_value = float(raw_val)
                                            except (ValueError, TypeError):
                                                pass # Keep ess_value as None
                                except (IndexError, KeyError):
                                     pass # Round/Country doesn't exist for this question

                            # Only add if *both* trade and ESS data are valid numbers for this year
                            if ess_value is not None:
                                question_timeseries.append({'Year': year, 'Trade': trade_quantity, 'ESS': ess_value})

                        # Check if enough data points for correlation
                        if len(question_timeseries) >= MIN_DATA_POINTS:
                            df_q = pd.DataFrame(question_timeseries)

                            # Check for constant values (no variation)
                            if df_q['Trade'].nunique() <= 1 or df_q['ESS'].nunique() <= 1:
                                continue # Skip if trade or ESS value is constant

                            try:
                                # Calculate Pearson correlation
                                correlation = df_q['Trade'].corr(df_q['ESS'], method='pearson')

                                # Check if correlation is valid
                                if pd.notna(correlation) and np.isfinite(correlation):
                                    all_correlations.append({
                                        'country': country_name,
                                        'question': question,
                                        'correlation': correlation,
                                        'data_points': len(df_q),
                                        'dataset': dataset_name,
                                        'trade_type': trade_type,
                                        'answer_type': answer_type
                                    })
                                    processed_count += 1
                                    if processed_count % 1000 == 0:
                                        print(f"      Processed {processed_count} potential correlations...")

                            except Exception as e:
                                # print(f"      Error calculating correlation for {country_name}, {question}: {e}")
                                pass # Silently ignore calculation errors for now

    print(f"Finished calculating correlations. Found {len(all_correlations)} valid correlations.")
    return all_correlations

def main(output_filename="all_correlations.csv"):
    """Main function to calculate correlations and save all results to a CSV."""
    print("Starting correlation analysis...")
    correlations = calculate_all_correlations()

    if not correlations:
        print("No valid correlations found.")
        return

    # Convert list of dictionaries to DataFrame
    df_correlations = pd.DataFrame(correlations)

    # Sort by absolute value of correlation, descending
    df_correlations['abs_correlation'] = df_correlations['correlation'].abs()
    df_sorted = df_correlations.sort_values(by='abs_correlation', ascending=False).drop(columns=['abs_correlation'])

    # Define the output path (relative to the script's location in src)
    # Go up one level from src to the project root
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', output_filename)

    try:
        # Save the DataFrame to CSV
        df_sorted.to_csv(output_path, index=False, float_format='%.5f')
        print(f"Successfully saved {len(df_sorted)} correlations to {output_path}")
    except Exception as e:
        print(f"Error saving correlations to CSV: {e}")

if __name__ == "__main__":
    main()