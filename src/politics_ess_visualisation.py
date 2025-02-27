from urllib.request import urlretrieve
import os
import zipfile
from pathlib import Path
import logging
import pandas
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as colors

from dash import Dash, dcc, html, Input, Output

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG
)


def loadDataframe():
    # the following dataset contains countries that took the survey across 2002 to 2023
    # i just manually picked the countries on the dataset builder page
    # i've also just selected the 'politics' variables (we can change this if needed)
    # url: https://ess.sikt.no/en/data-builder/?tab=round_country&rounds=0.2_9-11_13_15_23-38+1.2_9-11_13_15_23-27_32-5_38+2.2_9-27_32-35_38+3.2_9-11_13_15_23-27_32-35_38+4.2_9-11_13_15_23-27_32-35_38+5.2_9-13_15_23-27_32-35_38+6.2_9-15_23-38+7.2_9-13_15_23-27_32-38+8.2_9-13_15_23-27_32-38+9.2_9_10_13_15_23_25_27_32-38+10.11_26_33_34+11.2_9-11_13_15_23-27_32-38&seriesVersion=883&variables=1
    dataset = Path(
        "datasets/ESS1e06_7-ESS2e03_6-ESS3e03_7-ESS4e04_6-ESS5e03_5-ESS6e02_6-ESS7e02_3-ESS8e02_3-ESS9e03_2-ESS10-ESS10SC-ESS11-subset.csv"
    )

    if not dataset.is_file():
        logger.info("Dataset is not downloaded. Downloading now.")
        dataset_url = "https://stessdissprodwe.blob.core.windows.net/data/download/4/generate_datafile/714852f71208f80d1af68aefdacaab25.zip?st=2025-02-27T12%3A35%3A10Z&se=2025-02-27T13%3A37%3A10Z&sp=r&sv=2023-11-03&sr=b&skoid=1b26ad26-8999-4f74-9562-ad1c57749956&sktid=93a182aa-d7bd-4a74-9fb1-84df14cae517&skt=2025-02-27T12%3A35%3A10Z&ske=2025-02-27T13%3A37%3A10Z&sks=b&skv=2023-11-03&sig=93r9f2LuIpSZnoc59tBB65m9f7ohcKBEXRRBN68gH94%3D"
        dst = "datasets/ess_politics.zip"
        _ = urlretrieve(dataset_url, dst)

        # unzip and clean up
        with zipfile.ZipFile(dst, "r") as zip_ref:
            file_names = []
            files = zip_ref.infolist()
            zip_len = len(files)

            for i in range(zip_len):
                file_names.append(files[i].filename)

            zip_ref.extractall("datasets/")
            os.remove(dst)

            html_files = [file for file in file_names if "html" in file]
            for i in range(len(html_files)):
                os.remove(f"datasets/{html_files[0]}")

    return pandas.read_csv(dataset)

def create_comparison_figure(df, trust_metric):
    # Calculate average trust for all countries
    df_filtered = df[(df[trust_metric] >= 0) & (df[trust_metric] <= 10)].copy()
    df_filtered['trust_score'] = df_filtered[trust_metric].astype(float)
    average_trust = df_filtered.groupby(['essround', 'cntry'])['trust_score'].mean().reset_index()

    # Create figure for country comparison
    fig = go.Figure()

    # Use Viridis color scale
    num_countries = len(df['cntry'].unique())
    viridis_colors = colors.sequential.Viridis
    country_colors = [viridis_colors[i % len(viridis_colors)] for i in range(num_countries)]

    # Add a line for each country
    for i, country in enumerate(df['cntry'].unique()):
        country_data = average_trust[average_trust['cntry'] == country]
        country_name = country_name_map.get(country, country)
        fig.add_trace(go.Scatter(
            x=country_data['essround'],
            y=country_data['trust_score'],
            mode='lines+markers',
            name=country_name,
            hovertemplate=f"{country_name}, Round: %{{x}}, Trust: %{{y:.2f}}<extra></extra>",
            marker=dict(color=country_colors[i])  # Assign color to the line
        ))

    fig.update_layout(
        title=f"Average Trust in {trust_metric} Across All Countries",
        title_x=0.5,
        xaxis_title="ESS Round",
        yaxis_title="Average Trust Score",
        yaxis_range=[0, 10],
        xaxis=dict(tickmode='linear'),
        height=500,
        hovermode='x unified'
    )
    return fig

def create_detail_figure(df, selected_country, trust_metric):
    df_country = df[df['cntry'] == selected_country].copy()
    df_country = df_country[(df_country[trust_metric] >= 0) & (df_country[trust_metric] <= 10)]

    grouped = df_country.groupby(['essround', trust_metric]).size().unstack(fill_value=0)
    grouped_norm = grouped.apply(lambda x: x/x.sum(), axis=1)

    viridis_colors = colors.sequential.Viridis

    count_traces = []
    num_columns_grouped = len(grouped.columns)
    bar_colors_counts = [viridis_colors[i % len(viridis_colors)] for i in range(num_columns_grouped)]
    for i, column in enumerate(grouped.columns):
        trust_label = str(column)
        if column == 0:
            trust_label = "0 - No trust at all"
        elif column == 10:
            trust_label = "10 - Complete trust"
        count_traces.append(go.Bar(x=grouped.index, y=grouped[column], name=trust_label, marker_color=bar_colors_counts[i], legendgroup="group1"))

    proportion_traces = []
    num_columns_grouped_norm = len(grouped_norm.columns)
    bar_colors_props = [viridis_colors[i % len(viridis_colors)] for i in range(num_columns_grouped_norm)]
    for i, column in enumerate(grouped_norm.columns):
        trust_label = str(column)
        if column == 0:
            trust_label = "0 - No trust at all"
        elif column == 10:
            trust_label = "10 - Complete trust"
        proportion_traces.append(go.Bar(x=grouped_norm.index, y=grouped_norm[column], name=trust_label, marker_color=bar_colors_props[i], legendgroup="group1", showlegend=False))

    fig = make_subplots(rows=2, cols=1, subplot_titles=(f"Trust in {trust_metric} (Counts)", f"Normalised Trust in {trust_metric} (Proportions)"), vertical_spacing=0.2)

    for trace in count_traces:
        fig.add_trace(trace, row=1, col=1)
    for trace in proportion_traces:
        fig.add_trace(trace, row=2, col=1)

    fig.update_layout(
        title_text=f"Detailed Trust Distribution in {country_name_map.get(selected_country, selected_country)} - {trust_metric}",
        title_x=0.5,
        height=800,
        barmode='stack',
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1)
    )
    fig.update_yaxes(title_text="Number of Respondents", row=1, col=1)
    fig.update_yaxes(title_text="Proportion of Respondents", row=2, col=1)
    fig.update_xaxes(title_text="ESS Round", row=1, col=1, tickmode='linear')
    fig.update_xaxes(title_text="ESS Round", row=2, col=1, tickmode='linear')
    return fig

app = Dash(__name__)

df = loadDataframe()
logger.info("ESS Dataset loaded")

columns_to_drop = ["name", "edition","proddate"]
df = df.drop(columns=columns_to_drop, errors='ignore')
countries = df['cntry'].unique()

# Mapping of country codes to full country names
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

# Use the mapping to create dropdown options
country_dropdown_options = [{'label': country_name_map.get(country, country), 'value': country} for country in countries]

trust_metrics = {
    'trstep': 'Trust in the European Parliament',
    'trstlgl': 'Trust in the legal system',
    'trstplc': 'Trust in the police',
    'trstplt': 'Trust in politicians',
    'trstprl': 'Trust in country\'s parliament',
    'trstprt': 'Trust in political parties',
    'trstun': 'Trust in the United Nations',
    'trstsci': 'Trust in scientists'
}

trust_metric_options = [{'label': label, 'value': metric} for metric, label in trust_metrics.items()]

app.layout = html.Div([
    html.H1("Trust in Politicians Across Europe", style={'textAlign': 'center'}),

    # Comparison plot
    dcc.Graph(id='comparison-plot'),

    # Country selector
    dcc.Dropdown(
        id='country-dropdown',
        options=country_dropdown_options,
        value=countries[0],
        clearable=False,
        style={'width': '50%', 'margin': '20px auto'}
    ),
    # Trust Metric selector
    dcc.Dropdown(
        id='trust-metric-dropdown',
        options=trust_metric_options,
        value='trstplt',  # Default trust metric
        clearable=False,
        style={'width': '50%', 'margin': '20px auto'}
    ),

    # Detailed country plot
    dcc.Graph(id='detail-plot')
])

@app.callback(
    [Output('comparison-plot', 'figure'),
     Output('detail-plot', 'figure')],
    [Input('country-dropdown', 'value'),
     Input('trust-metric-dropdown', 'value')]
)
def update_graphs(selected_country, trust_metric):
    comparison_fig = create_comparison_figure(df, trust_metric)
    detail_fig = create_detail_figure(df, selected_country, trust_metric)
    return comparison_fig, detail_fig

if __name__ == '__main__':
    app.run_server(debug=True)

