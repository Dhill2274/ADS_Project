from urllib.request import urlretrieve
import os
import zipfile
from pathlib import Path
import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as colors
import plotly.express as px

from dash import Dash, dcc, html, Input, Output

# Map ESS rounds to the actual years
round_year_map = {
    1: 2002,
    2: 2004,
    3: 2006,
    4: 2008,
    5: 2010,
    6: 2012,
    7: 2014,
    8: 2016,
    9: 2018,
    10: 2020,
    11: 2023
}

ess_years = list(round_year_map.values())

# Country code → country name mapping
country_name_map = {
    'BE': 'Belgium',
    'FI': 'Finland',
    'FR': 'France',
    'DE': 'Germany',
    'HU': 'Hungary',
    'IE': 'Ireland',
    'NL': 'Netherlands',
    'NO': 'Norway',
    'PL': 'Poland',
    'PT': 'Portugal',
    'SI': 'Slovenia',
    'ES': 'Spain',
    'SE': 'Sweden',
    'CH': 'Switzerland',
    'GB': 'United Kingdom'
}

# For filtering, we only want these country names:
ess_country_names = set(country_name_map.values())


def get_average_importance_by_round(df, importance_metric):
    """
    Returns a DataFrame with columns:
      ['cntry', 'essround', 'year', 'avg_importance_score']
    where 'year' is mapped from essround.
    """
    # Filter valid values
    df_filtered = df[(df[importance_metric] >= 0) & (df[importance_metric] <= 10)].copy()
    df_filtered['importance_score'] = df_filtered[importance_metric].astype(float)

    # Group by country and ESS round, compute the average
    grouped = df_filtered.groupby(['cntry', 'essround'])['importance_score'].mean().reset_index()
    grouped.rename(columns={'importance_score': 'avg_importance_score'}, inplace=True)

    grouped['year'] = grouped['essround'].map(round_year_map)

    return grouped

def load_military_data(filepath="Datasets/Percentage_Share_of_GDP_Military_Expenditure_imputed.csv"):
    """
    Loads the placeholder dataset for military expenditure. 
    Must contain at least ['cntry', 'year', 'military_expenditure'].
    """
    mil_df = pd.read_csv(filepath)

    mil_df = mil_df[mil_df['Country'].isin(ess_country_names)]
    keep_cols = ['Country'] + [str(year) for year in ess_years]
    mil_df = mil_df[keep_cols]

    name_to_code = {v: k for k, v in country_name_map.items()}

    df_long = mil_df.melt(
        id_vars='Country',
        var_name='year',
        value_name='military_expenditure'
    )

    # Convert the 'year' column to numeric (it will be string after melt)
    df_long['year'] = df_long['year'].astype(int)

    # Map the full country name to its code
    df_long['cntry'] = df_long['Country'].map(name_to_code)

    # Optional: reorder columns for clarity
    df_long = df_long[['cntry', 'Country', 'year', 'military_expenditure']]

    return df_long

def merge_importance_and_military(avg_importance_df, military_df):
    merged = pd.merge(
        avg_importance_df,
        military_df,
        on=['cntry', 'year'],
        how='inner'  # or 'left' if you want to keep unmatched rows
    )
    return merged

def create_correlation_plot(merged_df, selected_country):
    """
    Create a scatter plot showing correlation between 
    avg_importance_score and military_expenditure 
    for a given country across ESS years.
    """
    # Filter data for the chosen country
    country_data = merged_df[merged_df['cntry'] == selected_country].copy()
    
    # Compute correlation coefficient
    corr_matrix = country_data[['avg_importance_score', 'military_expenditure']].corr()
    corr_value = corr_matrix.iloc[0,1]  # correlation between the two columns
    
    # Build the scatter plot with a trendline
    fig = px.scatter(
        country_data,
        x='avg_importance_score',
        y='military_expenditure',
        trendline='ols',  # ordinary least squares trendline
        labels={
            'avg_importance_score': 'Average Importance Score',
            'military_expenditure': 'Military Expenditure (% of GDP?)'
        },
        title=f"Correlation Plot: {selected_country} (r={corr_value:.2f})"
    )

    return fig

app = Dash(__name__)

if __name__ == '__main__':
    dataset = Path("Cleaned/ESS-Human-Values-Cleaned.csv")
    data_ESS = pd.read_csv(dataset)

    new_ESS = get_average_importance_by_round(data_ESS, 'impdiffa')
    milit_data = load_military_data()
    merged = merge_importance_and_military(new_ESS, milit_data)

    fig = create_correlation_plot(merged, 'GB')
    
    # Instead of px.imshow(fig), just call:
    fig.show()
