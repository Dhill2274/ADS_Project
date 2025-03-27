import pandas as pd
import numpy as np
import json
from collections import defaultdict
import concurrent.futures
import time

def load_raw_data():
    """
    Load the raw trade data from the cleaned CSV file
    """
    path = "cleaned/arms/trades-clean.csv"
    df = pd.read_csv(path)
    return df

def create_yearly_aggregates(df):
    """
    Create yearly aggregates of trade data (total volume by year)
    """
    # Convert year to integer and drop rows with missing year
    df['Year of order'] = pd.to_numeric(df['Year of order'], errors='coerce')
    df = df.dropna(subset=['Year of order'])
    df['Year of order'] = df['Year of order'].astype(int)

    # Group by year and calculate metrics
    yearly_agg = df.groupby('Year of order').agg(
        total_trades=('Quantity', 'count'),
        total_quantity=('Quantity', 'sum'),
        avg_quantity=('Quantity', 'mean'),
        max_quantity=('Quantity', 'max')
    ).reset_index()

    return yearly_agg

def create_country_aggregates(df):
    """
    Create country-level aggregates for both exporters and importers
    """
    # Filter to only include rows with positive quantities
    df = df[df['Quantity'] > 0]

    # Convert year to integer and drop rows with missing year
    df['Year of order'] = pd.to_numeric(df['Year of order'], errors='coerce')
    df = df.dropna(subset=['Year of order'])
    df['Year of order'] = df['Year of order'].astype(int)

    # Group by exporter (From) and calculate metrics
    exporter_agg = df.groupby('From').agg(
        export_trades=('Quantity', 'count'),
        export_quantity=('Quantity', 'sum'),
        export_avg_quantity=('Quantity', 'mean'),
        export_max_quantity=('Quantity', 'max'),
        export_countries=('To', 'nunique')
    ).reset_index()

    # Group by importer (To) and calculate metrics
    importer_agg = df.groupby('To').agg(
        import_trades=('Quantity', 'count'),
        import_quantity=('Quantity', 'sum'),
        import_avg_quantity=('Quantity', 'mean'),
        import_max_quantity=('Quantity', 'max'),
        import_countries=('From', 'nunique')
    ).reset_index()

    return exporter_agg, importer_agg

def create_country_year_aggregates(df):
    """
    Create country-year aggregates (quantities by country and year)
    """
    # Filter to only include rows with positive quantities
    df = df[df['Quantity'] > 0]

    # Convert year to integer and drop rows with missing year
    df['Year of order'] = pd.to_numeric(df['Year of order'], errors='coerce')
    df = df.dropna(subset=['Year of order'])
    df['Year of order'] = df['Year of order'].astype(int)

    # Group by exporter and year
    exporter_year_agg = df.groupby(['From', 'Year of order']).agg(
        export_trades=('Quantity', 'count'),
        export_quantity=('Quantity', 'sum')
    ).reset_index()

    # Group by importer and year
    importer_year_agg = df.groupby(['To', 'Year of order']).agg(
        import_trades=('Quantity', 'count'),
        import_quantity=('Quantity', 'sum')
    ).reset_index()

    return exporter_year_agg, importer_year_agg

def get_top_trading_pairs(df, n=20):
    """
    Identify the top trading pairs (from-to) based on quantity
    """
    # Group by From and To
    pair_agg = df.groupby(['From', 'To']).agg(
        trade_count=('Quantity', 'count'),
        total_quantity=('Quantity', 'sum')
    ).reset_index()

    # Sort by total quantity and get top N
    top_pairs = pair_agg.sort_values('total_quantity', ascending=False).head(n)

    return top_pairs

def create_weapon_category_aggregates(df):
    """
    Create aggregates by weapon category
    """
    # Group by weapon description
    weapon_agg = df.groupby('Weapon description').agg(
        trade_count=('Quantity', 'count'),
        total_quantity=('Quantity', 'sum'),
        avg_quantity=('Quantity', 'mean')
    ).reset_index()

    # Sort by total quantity
    weapon_agg = weapon_agg.sort_values('total_quantity', ascending=False)

    return weapon_agg

def save_aggregates(file_prefix="cleaned/arms/"):
    """
    Generate and save all aggregated data
    """
    print("Loading raw data...")
    df = load_raw_data()

    print("Creating yearly aggregates...")
    yearly_agg = create_yearly_aggregates(df)
    yearly_agg.to_csv(f"{file_prefix}yearly_aggregates.csv", index=False)

    print("Creating country aggregates...")
    exporter_agg, importer_agg = create_country_aggregates(df)
    exporter_agg.to_csv(f"{file_prefix}exporter_aggregates.csv", index=False)
    importer_agg.to_csv(f"{file_prefix}importer_aggregates.csv", index=False)

    print("Creating country-year aggregates...")
    exporter_year_agg, importer_year_agg = create_country_year_aggregates(df)
    exporter_year_agg.to_csv(f"{file_prefix}exporter_year_aggregates.csv", index=False)
    importer_year_agg.to_csv(f"{file_prefix}importer_year_aggregates.csv", index=False)

    print("Identifying top trading pairs...")
    top_pairs = get_top_trading_pairs(df)
    top_pairs.to_csv(f"{file_prefix}top_trading_pairs.csv", index=False)

    print("Creating weapon category aggregates...")
    weapon_agg = create_weapon_category_aggregates(df)
    weapon_agg.to_csv(f"{file_prefix}weapon_aggregates.csv", index=False)

    print("All aggregates saved!")

def run_preprocessing():
    """
    Run all preprocessing steps
    """
    start_time = time.time()
    save_aggregates()
    end_time = time.time()
    print(f"Preprocessing completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    run_preprocessing()