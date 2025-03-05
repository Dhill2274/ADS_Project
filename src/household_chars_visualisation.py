from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import plotly.colors as colors

# Mapping of country codes to full country names; only these countries will appear in the dropdown and comparison graph
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

def load_household_data():
    dataset = Path("Datasets/cleaned_household_characteristics.csv")
    if not dataset.is_file():
        raise FileNotFoundError("Dataset not found at 'Datasets/cleaned_household_characteristics.csv'")
    return pd.read_csv(dataset)

def create_household_distribution_figure(df, selected_country='all'):
    """
    Creates a bar chart showing the frequency distribution of household sizes on a linear y-axis.
    If selected_country is not 'all', it filters the data to that country.
    The x-axis is fixed to range from 0 to 24.
    """
    df_filtered = df[df['hhmmb'].notnull() & (df['hhmmb'] > 0)]
    if selected_country != 'all':
        df_filtered = df_filtered[df_filtered['cntry'] == selected_country]
    
    counts = df_filtered['hhmmb'].value_counts().sort_index().reset_index()
    counts.columns = ['hhmmb', 'count']
    
    title_suffix = ""
    if selected_country != 'all':
        title_suffix = f" in {country_name_map.get(selected_country, selected_country)}"
    
    fig = go.Figure(
        data=[go.Bar(
            x=counts['hhmmb'],
            y=counts['count'],
            marker_color='teal'
        )]
    )
    fig.update_layout(
        title="Distribution of Number of People per Household" + title_suffix,
        xaxis_title="Number of People per Household",
        yaxis_title="Frequency",
        xaxis=dict(tickmode='linear', range=[0, 24])
    )
    return fig

def create_household_distribution_log_figure(df, selected_country='all'):
    """
    Creates a bar chart showing the frequency distribution of household sizes on a log-scaled y-axis.
    If selected_country is not 'all', it filters the data to that country.
    The x-axis is fixed to range from 0 to 24.
    """
    df_filtered = df[df['hhmmb'].notnull() & (df['hhmmb'] > 0)]
    if selected_country != 'all':
        df_filtered = df_filtered[df_filtered['cntry'] == selected_country]
    
    counts = df_filtered['hhmmb'].value_counts().sort_index().reset_index()
    counts.columns = ['hhmmb', 'count']
    
    title_suffix = ""
    if selected_country != 'all':
        title_suffix = f" in {country_name_map.get(selected_country, selected_country)}"
    
    fig = go.Figure(
        data=[go.Bar(
            x=counts['hhmmb'],
            y=counts['count'],
            marker_color='teal'
        )]
    )
    fig.update_layout(
        title="Distribution of Number of People per Household (Log Scale)" + title_suffix,
        xaxis_title="Number of People per Household",
        yaxis_title="Frequency (log scale)",
        xaxis=dict(tickmode='linear', range=[0, 24]),
        yaxis=dict(type="log")
    )
    return fig

def create_comparison_figure(df):
    """
    Creates a line chart showing the average number of people per household
    by ESS round for each country in country_name_map.
    """
    df_filtered = df[df['hhmmb'].notnull() & (df['hhmmb'] > 0)]
    df_filtered = df_filtered[df_filtered['cntry'].isin(country_name_map.keys())]
    avg_household = df_filtered.groupby(['essround', 'cntry'])['hhmmb'].mean().reset_index()
    
    fig = go.Figure()
    unique_countries = sorted(df_filtered['cntry'].unique())
    num_countries = len(unique_countries)
    viridis_colors = colors.sequential.Viridis
    country_colors = [viridis_colors[i % len(viridis_colors)] for i in range(num_countries)]
    
    for i, country in enumerate(unique_countries):
        country_data = avg_household[avg_household['cntry'] == country]
        country_name = country_name_map.get(country, country)
        fig.add_trace(go.Scatter(
            x=country_data['essround'],
            y=country_data['hhmmb'],
            mode='lines+markers',
            name=country_name,
            hovertemplate=f"{country_name}, ESS Round: %{{x}}, Avg Household Size: %{{y:.2f}}<extra></extra>",
            marker=dict(color=country_colors[i])
        ))
    
    fig.update_layout(
        title="Average Number of People per Household by ESS Round per Country",
        xaxis_title="ESS Round",
        yaxis_title="Average Number of People per Household",
        xaxis=dict(tickmode='linear'),
        hovermode='x unified',
        height=500
    )
    return fig

# Initialize the Dash app
app = Dash(__name__)

# Load data
df_household = load_household_data()

# Create initial figures
initial_distribution_fig = create_household_distribution_figure(df_household, selected_country='all')
initial_distribution_log_fig = create_household_distribution_log_figure(df_household, selected_country='all')
comparison_fig = create_comparison_figure(df_household)

# Define options for the distribution graph dropdown
distribution_country_options = (
    [{'label': 'All Countries', 'value': 'all'}] +
    [{'label': country_name_map.get(country, country), 'value': country} for country in country_name_map.keys()]
)

# Define layout with three graphs:
# 1. Comparison graph (average by ESS round)
# 2. Distribution graph (linear scale) with dropdown
# 3. Distribution graph (log scale) with the same dropdown
app.layout = html.Div([
    html.H1("Household Characteristics Visualisation", style={'textAlign': 'center'}),
    
    # Comparison graph
    dcc.Graph(id='comparison-graph', figure=comparison_fig),
    
    # Dropdown for distribution graphs
    html.Div([
        dcc.Dropdown(
            id='distribution-country-dropdown',
            options=distribution_country_options,
            value='all',
            clearable=False,
            style={'width': '50%', 'margin': '20px auto'}
        )
    ]),
    
    # Distribution graph (linear scale)
    dcc.Graph(id='distribution-graph', figure=initial_distribution_fig),
    
    # Distribution graph (log scale)
    dcc.Graph(id='distribution-log-graph', figure=initial_distribution_log_fig), 

    # Extra space at the bottom to prevent final graph being blocked
    html.Div(style={'height': '50px'})
])

# Callback to update the linear distribution graph based on the selected country
@app.callback(
    Output('distribution-graph', 'figure'),
    Input('distribution-country-dropdown', 'value')
)
def update_distribution(selected_country):
    return create_household_distribution_figure(df_household, selected_country)

# Callback to update the log scale distribution graph based on the selected country
@app.callback(
    Output('distribution-log-graph', 'figure'),
    Input('distribution-country-dropdown', 'value')
)
def update_distribution_log(selected_country):
    return create_household_distribution_log_figure(df_household, selected_country)

if __name__ == '__main__':
    app.run_server(debug=True)
