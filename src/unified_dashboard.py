import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, callback, Input, Output, State, page_container
import dash_bootstrap_components as dbc
import json
import sys
import os
import numpy as np

# Add the parent directory to the path so we can import from the other directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from ESS module
from src.ess.dataset import Dataset, DATASET_DISPLAY_NAMES

# Constants from ESS module
ESS_COUNTRIES = {
    "AT": "Austria",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "CH": "Switzerland",
    "CY": "Cyprus",
    "CZ": "Czechia",
    "DE": "Germany",
    "DK": "Denmark",
    "EE": "Estonia",
    "ES": "Spain",
    "FI": "Finland",
    "FR": "France",
    "GB": "United Kingdom",
    "HU": "Hungary",
    "IE": "Ireland",
    "IL": "Israel",
    "LT": "Lithuania",
    "NL": "Netherlands",
    "NO": "Norway",
    "PL": "Poland",
    "PT": "Portugal",
    "RU": "Russian Federation",
    "SE": "Sweden",
    "SI": "Slovenia",
    "SK": "Slovakia",
    "UA": "Ukraine"
}

ESS_DATASETS = ["chars", "media", "politics", "socio", "values", "wellbeing"]

# List of ESS countries for trade visualization
ESS_COUNTRY_LIST = [
    "Austria", "Belgium", "Bulgaria", "Switzerland", "Cyprus", "Czechia",
    "Germany", "Denmark", "Estonia", "Spain", "Finland", "France",
    "United Kingdom", "Hungary", "Ireland", "Israel", "Lithuania",
    "Netherlands", "Norway", "Poland", "Portugal", "Russian Federation",
    "Sweden", "Slovenia", "Slovakia", "Ukraine"
]

# Create the Dash app with Bootstrap
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Unified Data Visualisation Dashboard",
    suppress_callback_exceptions=True
)

# Create the navbar
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("Data Visualisation Dashboard", className="ms-2"),
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("ESS Visualisation", href="#ess", id="ess-link")),
                    dbc.NavItem(dbc.NavLink("Trade Quantity", href="#trade-quantity", id="trade-quantity-link")),
                    dbc.NavItem(dbc.NavLink("Correlation Matrix", href="#correlation-matrix", id="correlation-matrix-link")),
                ],
                className="ms-auto",
                navbar=True,
            ),
        ]
    ),
    color="primary",
    dark=True,
    className="mb-4",
)

# Create the app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content'),
])

###################
# ESS VISUALISATION
###################

# ESS visualisation layout
ess_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("European Social Survey Data Visualisation",
                   className="text-center mb-4 mt-3"),
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Data Selection"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label('Select Dataset:', className="fw-bold"),
                            dcc.Dropdown(
                                id='ess-dataset-dropdown',
                                options=[{'label': DATASET_DISPLAY_NAMES.get(dataset, dataset.capitalize()),
                                          'value': dataset}
                                         for dataset in ESS_DATASETS],
                                value=ESS_DATASETS[0],
                                clearable=False,
                                className="mb-3"
                            ),
                        ], xs=12, md=6),

                        dbc.Col([
                            html.Label('Select Question:', className="fw-bold"),
                            dcc.Dropdown(
                                id='ess-question-dropdown',
                                options=[],
                                value='',
                                placeholder="Select a question",
                                className="mb-3"
                            ),
                        ], xs=12, md=6),
                    ]),

                    dbc.Row([
                        dbc.Col([
                            html.Label('Visualisation Type:', className="fw-bold"),
                            dbc.RadioItems(
                                id='ess-visualization-type',
                                options=[
                                    {'label': 'Most Common Answer', 'value': 'mode'},
                                    {'label': 'Mean Answer', 'value': 'mean'}
                                ],
                                value='mean',
                                inline=True,
                                className="mt-1 mb-3"
                            ),
                        ], xs=12),
                    ]),

                    # Help text about legend filtering
                    dbc.Row([
                        dbc.Col([
                            html.P("Tip: Double click on country names in the graph legend to hide or show specific countries.",
                                  className="text-muted font-italic small")
                        ])
                    ])
                ])
            ], className="mb-4 shadow-sm"),
        ], xs=12)
    ]),

    dbc.Card([
        dbc.CardBody([
            dcc.Loading(
                id="ess-loading-graph",
                type="circle",
                children=[
                    dcc.Graph(
                        id='ess-survey-graph',
                        style={'height': '600px'},
                        figure=go.Figure(layout={
                            'title': 'Select a dataset and question to view data',
                            'xaxis': {'title': 'ESS Round', 'range': [0.5, 11.5], 'dtick': 1},
                            'yaxis': {'title': 'Response Value', 'range': [1, None]},
                            'template': 'plotly_white'
                        })
                    )
                ]
            )
        ])
    ], className="mb-4 shadow-sm"),

    dbc.Card([
        dbc.CardHeader("Dataset Information"),
        dbc.CardBody(id='ess-dataset-info')
    ], className="shadow-sm"),

    html.Footer([
        html.P("European Social Survey Data Visualisation Tool", className="text-center text-muted mt-4")
    ])
], fluid=True, className="px-4 py-3")

# ESS callbacks
@callback(
    [Output('ess-question-dropdown', 'options'),
     Output('ess-question-dropdown', 'value'),
     Output('ess-dataset-info', 'children')],
    Input('ess-dataset-dropdown', 'value')
)
def update_ess_question_dropdown(selected_dataset):
    if not selected_dataset:
        return [], '', html.P("Please select a dataset")

    try:
        dataset_obj = Dataset(selected_dataset)
        # Show all questions instead of limiting to first 15
        question_options = [{'label': q[:80] + '...' if len(q) > 80 else q,
                            'value': q}
                           for q in dataset_obj.questionLabels]

        # Set default question value
        default_question = dataset_obj.questionLabels[0] if dataset_obj.questionLabels else ''

        # Use the display name in the dataset info
        display_name = DATASET_DISPLAY_NAMES.get(selected_dataset, selected_dataset.capitalize())

        dataset_info = html.Div([
            html.H4(f"Dataset: {display_name}"),
            html.P(f"Available countries: {len(dataset_obj.countryLabels)}"),
            html.P(f"Available rounds: {', '.join(map(str, dataset_obj.rounds))}")
        ])

        return question_options, default_question, dataset_info

    except Exception as e:
        return [], '', html.P(f"Error loading dataset: {str(e)}")

@callback(
    Output('ess-survey-graph', 'figure'),
    [Input('ess-dataset-dropdown', 'value'),
     Input('ess-question-dropdown', 'value'),
     Input('ess-visualization-type', 'value')]
)
def update_ess_graph(selected_dataset, selected_question, viz_type):
    if not selected_dataset or not selected_question:
        # Return empty figure with instructions
        return go.Figure(layout={
            'title': 'Select a dataset and question to view data',
            'xaxis': {'title': 'ESS Round', 'range': [0.5, 11.5], 'dtick': 1},
            'yaxis': {'title': 'Response Value', 'range': [1, None]},
            'template': 'plotly_white'
        })

    try:
        dataset_obj = Dataset(selected_dataset)
        fig = go.Figure()

        rounds = list(range(1, 12))

        # This now triggers lazy loading just for the selected question
        # Choose data source based on visualization type
        data_source = dataset_obj.questions if viz_type == 'mode' else dataset_obj.questionsMean
        question_data = data_source[selected_question]  # This triggers the actual processing

        viz_title = "Most Common Answer" if viz_type == 'mode' else "Mean Answer"
        display_name = DATASET_DISPLAY_NAMES.get(selected_dataset, selected_dataset.capitalize())

        # Always show all countries
        for country_code, country_name in ESS_COUNTRIES.items():
            if country_code in dataset_obj.countryLabels:
                y_values = question_data[country_code]

                # Convert answers to numerical values if possible for better visualisation
                try:
                    y_values_numeric = [float(y) if y != "None" else None for y in y_values]
                except (ValueError, TypeError):
                    # If conversion fails, keep as categorical values
                    y_values_numeric = y_values

                fig.add_trace(
                    go.Scatter(
                        x=rounds,
                        y=y_values_numeric,
                        mode='lines+markers',
                        name=country_name,
                        hovertemplate=f"Country: {country_name}<br>Round: %{{x}}<br>Answer: %{{y}}<extra></extra>"
                    )
                )

        # Update layout with display name
        fig.update_layout(
            title=f"{viz_title} for: {selected_question[:50] + '...' if len(selected_question) > 50 else selected_question} ({display_name})",
            xaxis_title="ESS Round",
            yaxis_title="Response Value",
            legend_title="Countries",
            hovermode="closest",
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            xaxis=dict(range=[0.5, 11.5], dtick=1, tickmode='linear'),
            yaxis=dict(range=[1, None]),
            template='plotly_white'
        )

        return fig

    except Exception as e:
        # Return error figure
        return go.Figure(layout={
            'title': f"Error: {str(e)}",
            'xaxis': {'title': 'ESS Round', 'range': [0.5, 11.5], 'dtick': 1},
            'yaxis': {'title': 'Response Value', 'range': [1, None]},
            'template': 'plotly_white'
        })

#########################
# TRADE QUANTITY VISUALISATION
#########################

# Load and prepare the trade data
def load_trade_data():
    print("Loading trade data...")
    path = "cleaned/arms/trades-clean.csv"
    df = pd.read_csv(path)

    # Filter to only include rows with positive quantities
    df = df[df['Quantity'] > 0]

    # Convert year to integer
    df['Year of order'] = pd.to_numeric(df['Year of order'], errors='coerce')
    df = df.dropna(subset=['Year of order'])
    df['Year of order'] = df['Year of order'].astype(int)

    # Load country coordinates for reference
    with open('src/trade-weapon-world-visualisation/country_coordinates.json', 'r') as f:
        country_coords = json.load(f)

    print(f"Loaded {len(df)} trade records")
    return df, country_coords

# Process data for quantity visualization
def process_quantity_data(df):
    print("Processing quantity data...")

    # Get list of unique years
    years = sorted(df['Year of order'].unique())

    # Get list of all unique countries (both exporters and importers)
    exporters = df['From'].unique()
    importers = df['To'].unique()
    all_countries = sorted(set(list(exporters) + list(importers)))

    # Create data structures for storing quantity information
    sent_per_country = {}  # Quantity sent from each country by year
    received_per_country = {}  # Quantity received by each country by year

    # Initialize data structures
    for country in all_countries:
        sent_per_country[country] = {year: 0 for year in years}
        received_per_country[country] = {year: 0 for year in years}

    # Populate data structures with quantity information
    for _, row in df.iterrows():
        year = int(row['Year of order'])
        from_country = row['From']
        to_country = row['To']
        quantity = row['Quantity']

        # Add to the sending country's total for that year
        sent_per_country[from_country][year] += quantity

        # Add to the receiving country's total for that year
        received_per_country[to_country][year] += quantity

    print(f"Processed data for {len(all_countries)} countries across {len(years)} years")
    return years, all_countries, sent_per_country, received_per_country

# Trade quantity visualisation layout
trade_quantity_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Global Arms Trade Quantity Visualisation",
                   className="text-center mb-4 mt-3"),
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Data Selection"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label('View Type:', className="fw-bold"),
                            dbc.RadioItems(
                                id='trade-view-type',
                                options=[
                                    {'label': 'Weapons Sent (Exports)', 'value': 'sent'},
                                    {'label': 'Weapons Received (Imports)', 'value': 'received'}
                                ],
                                value='sent',
                                inline=True,
                                className="mb-3"
                            ),
                        ], xs=12, md=6),

                        dbc.Col([
                            html.Label('Countries to Display:', className="fw-bold"),
                            dbc.RadioItems(
                                id='trade-country-filter',
                                options=[
                                    {'label': 'All Countries', 'value': 'all'},
                                    {'label': 'ESS Countries Only', 'value': 'ess'}
                                ],
                                value='ess',
                                inline=True,
                                className="mb-3"
                            ),
                        ], xs=12, md=6),
                    ]),

                    # Help text
                    dbc.Row([
                        dbc.Col([
                            html.P("Tip: Use the range slider below the graph to zoom into specific time periods. Click on country names in the legend to hide or show specific countries.",
                                  className="text-muted font-italic small")
                        ])
                    ])
                ])
            ], className="mb-4 shadow-sm"),
        ], xs=12)
    ]),

    dbc.Card([
        dbc.CardBody([
            dcc.Loading(
                id="trade-loading-graph",
                type="circle",
                children=[
                    dcc.Graph(
                        id='trade-quantity-graph',
                        style={'height': '800px'},
                        figure=go.Figure(layout={
                            'title': 'Select parameters to view arms trade quantity data',
                            'xaxis': {
                                'title': 'Year',
                                'rangeslider': {'visible': True},
                                'range': [2002, 2023],
                                'dtick': 4,
                                'tickmode': 'linear'
                            },
                            'yaxis': {'title': 'Quantity of Arms'},
                            'template': 'plotly_white'
                        })
                    )
                ]
            )
        ])
    ], className="mb-4 shadow-sm"),

    dbc.Card([
        dbc.CardHeader("Trade Volume Information"),
        dbc.CardBody(id='trade-info')
    ], className="shadow-sm"),

    html.Footer([
        html.P("Global Arms Trade Quantity Visualisation Tool", className="text-center text-muted mt-4")
    ])
], fluid=True, className="px-4 py-3")

# Trade quantity callbacks
@callback(
    [Output('trade-quantity-graph', 'figure'),
     Output('trade-info', 'children')],
    [Input('trade-view-type', 'value'),
     Input('trade-country-filter', 'value')]
)
def update_trade_graph(view_type, country_filter):
    # Load and process data
    df, _ = load_trade_data()
    years, all_countries, sent_per_country, received_per_country = process_quantity_data(df)

    # Determine which data to show based on view type
    if view_type == 'sent':
        data_source = sent_per_country
        title_prefix = "Arms Exporting Countries"
        action_text = "Exported"
    else:
        data_source = received_per_country
        title_prefix = "Arms Importing Countries"
        action_text = "Imported"

    # Calculate total quantity over the entire time period for each country
    country_totals = {}
    for country in all_countries:
        total = sum(data_source[country][year] for year in years if year in data_source[country])
        country_totals[country] = total

    # Determine which countries to display based on filter
    countries_with_data = [country for country, total in country_totals.items() if total > 0]

    if country_filter == 'ess':
        # Filter to show only ESS countries
        countries_to_display = [country for country in countries_with_data if country in ESS_COUNTRY_LIST]
        title_prefix = f"ESS {title_prefix}"

        # If no ESS countries have data, show a message
        if not countries_to_display:
            fig = go.Figure()
            fig.update_layout(
                title="No ESS countries found in the selected data",
                xaxis_title="Year",
                yaxis_title=f"Quantity of Arms {action_text}",
                template='plotly_white'
            )

            info_content = html.Div([
                html.H4("No Data Available"),
                html.P("No ESS countries have data for the selected view type.")
            ])

            return fig, info_content
    else:
        # Show all countries with data, but limit to top 50 if there are too many
        if len(countries_with_data) > 50:
            top_countries = sorted(
                [(country, country_totals[country]) for country in countries_with_data],
                key=lambda x: x[1],
                reverse=True
            )[:50]
            countries_to_display = [country for country, _ in top_countries]
            title_prefix = f"Top 50 {title_prefix} (out of {len(countries_with_data)})"
        else:
            countries_to_display = countries_with_data
            title_prefix = f"All {title_prefix} ({len(countries_with_data)})"

    # Create the figure using Plotly Express for easier handling of multiple lines
    # First create a DataFrame from our data
    plot_data = []

    for country in countries_to_display:
        for year in years:
            quantity = data_source[country].get(year, 0)
            if quantity > 0:  # Only include points with non-zero values
                plot_data.append({
                    'Country': country,
                    'Year': year,
                    'Quantity': quantity
                })

    plot_df = pd.DataFrame(plot_data)

    # Create the line chart with Plotly Express
    fig = px.line(
        plot_df,
        x='Year',
        y='Quantity',
        color='Country',
        title=f"{title_prefix} (Over All Time)",
        labels={'Quantity': f'Quantity of Arms {action_text}', 'Year': 'Year'},
        template='plotly_white'
    )

    # Add markers to the lines
    fig.update_traces(mode='lines+markers')

    # Add a rangeslider to the x-axis and set default range
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeslider_thickness=0.07,  # Adjust thickness
        tickangle=0,
        range=[2002, 2023],  # Set default visible range
        dtick=4,  # Show tick marks every 4 years
        tickmode='linear'
    )

    # Update layout
    fig.update_layout(
        hovermode="closest",
        legend_title="Countries",
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    # Prepare information panel content
    total_quantity = sum(country_totals.values())
    displayed_countries_quantity = sum(country_totals[country] for country in countries_to_display)
    percentage = (displayed_countries_quantity / total_quantity) * 100 if total_quantity > 0 else 0

    info_content = html.Div([
        html.H4(f"Arms Trade Volume (All Time)"),
        html.P(f"Total quantity of arms {action_text.lower()}: {int(total_quantity):,}"),
        html.P(f"The displayed countries account for {percentage:.1f}% of all arms {action_text.lower()}"),
        html.P(f"Number of countries in dataset: {len(countries_with_data)}")
    ])

    return fig, info_content

#########################
# CORRELATION MATRIX VISUALISATION
#########################

# Correlation matrix visualisation layout
correlation_matrix_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Trade-ESS Data Correlation Analysis",
                   className="text-center mb-4 mt-3"),
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Dataset Selection"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label('Select ESS Dataset:', className="fw-bold"),
                            dcc.Dropdown(
                                id='corr-dataset-dropdown',
                                options=[{'label': DATASET_DISPLAY_NAMES.get(dataset, dataset.capitalize()),
                                          'value': dataset}
                                         for dataset in ESS_DATASETS],
                                value='values',
                                clearable=False,
                                className="mb-3"
                            ),
                        ], width=6),

                        dbc.Col([
                            html.Label('Trade Data Type:', className="fw-bold"),
                            dbc.RadioItems(
                                id='corr-trade-type',
                                options=[
                                    {'label': 'Weapons Exported', 'value': 'sent'},
                                    {'label': 'Weapons Imported', 'value': 'received'}
                                ],
                                value='sent',
                                inline=True,
                                className="mb-3"
                            ),
                        ], width=6),
                    ]),

                    dbc.Row([
                        dbc.Col([
                            html.Label('Answer Type:', className="fw-bold"),
                            dbc.RadioItems(
                                id='corr-answer-type',
                                options=[
                                    {'label': 'Mean Answers', 'value': 'mean'},
                                    {'label': 'Most Common Answers', 'value': 'mode'}
                                ],
                                value='mean',
                                inline=True,
                                className="mb-3"
                            ),
                        ], width=12),
                    ]),

                    # Help text
                    dbc.Row([
                        dbc.Col([
                            html.P("This visualisation shows correlations between trade quantities and ESS questions.",
                                  className="text-muted font-italic small")
                        ])
                    ])
                ])
            ], className="mb-4 shadow-sm"),
        ], xs=12)
    ]),

    dbc.Card([
        dbc.CardBody([
            dcc.Loading(
                id="corr-loading-graph",
                type="circle",
                children=[
                    dcc.Graph(
                        id='correlation-matrix-graph',
                        style={'height': '800px'},
                        figure=go.Figure(layout={
                            'title': 'Select datasets to view correlation matrix',
                            'template': 'plotly_white'
                        })
                    )
                ]
            )
        ])
    ], className="mb-4 shadow-sm"),

    dbc.Card([
        dbc.CardHeader("Correlation Information"),
        dbc.CardBody(id='corr-matrix-info')
    ], className="shadow-sm"),

    html.Footer([
        html.P("Trade-ESS Correlation Analysis Tool", className="text-center text-muted mt-4")
    ])
], fluid=True, className="px-4 py-3")

# Function to map ESS survey rounds to years
def map_round_to_year(round_num):
    # Approximate mapping of ESS rounds to years
    mapping = {
        1: 2002, 2: 2004, 3: 2006, 4: 2008, 5: 2010,
        6: 2012, 7: 2014, 8: 2016, 9: 2018, 10: 2020, 11: 2022
    }
    return mapping.get(round_num, None)

# Correlation matrix callbacks
@callback(
    Output('correlation-matrix-graph', 'figure'),
    Output('corr-matrix-info', 'children'),
    [Input('corr-dataset-dropdown', 'value'),
     Input('corr-trade-type', 'value'),
     Input('corr-answer-type', 'value')]
)
def update_correlation_matrix(selected_dataset, trade_type, answer_type):
    if not selected_dataset:
        # Return empty figure with instructions
        return go.Figure(layout={
            'title': 'Select datasets to view correlation matrix',
            'template': 'plotly_white'
        }), html.P("Please select a dataset")

    try:
        # Load the ESS dataset
        dataset_obj = Dataset(selected_dataset)
        display_name = DATASET_DISPLAY_NAMES.get(selected_dataset, selected_dataset.capitalize())

        # Load and process trade data
        trade_df, _ = load_trade_data()
        years, all_countries, sent_per_country, received_per_country = process_quantity_data(trade_df)

        # Determine which trade data to use based on selection
        trade_data = sent_per_country if trade_type == 'sent' else received_per_country
        trade_action = "Export" if trade_type == 'sent' else "Import"

        # Get the answer data source based on selection
        answer_data = dataset_obj.questions if answer_type == 'mode' else dataset_obj.questionsMean
        answer_type_text = "Most Common Answers" if answer_type == 'mode' else "Mean Answers"

        # Get ESS countries that have trade data
        ess_country_codes = list(ESS_COUNTRIES.keys())

        # Create a mapping between ESS country codes and full names
        code_to_name = {code: name for code, name in ESS_COUNTRIES.items()}
        name_to_code = {name: code for code, name in ESS_COUNTRIES.items()}

        # Get top questions (limit to 15 for readability)
        questions = dataset_obj.questionLabels[:15]

        # Dictionary to store correlation values indexed by country and question
        country_question_correlations = {}

        # Countries and questions that have sufficient data for correlation
        valid_countries = []
        valid_questions = []

        # Process each country
        for country_code in ess_country_codes:
            if country_code not in dataset_obj.countryLabels:
                continue

            country_name = code_to_name[country_code]

            # Check if this country has trade data
            if country_name not in trade_data:
                continue

            # Create time series data for this country
            country_data = []
            country_has_sufficient_data = False

            # For each ESS round/year
            for round_num in range(1, 12):  # ESS rounds 1-11
                year = map_round_to_year(round_num)
                if year is None or year not in trade_data[country_name]:
                    continue

                # Get trade quantity for this country and year
                trade_quantity = trade_data[country_name][year]

                # Dictionary to hold question values for this year
                year_data = {'Year': year, 'Trade_Quantity': trade_quantity}

                # Add ESS question values
                has_question_data = False
                for question in questions:
                    if country_code in answer_data[question]:
                        value = answer_data[question][country_code][round_num - 1]  # Adjust for 0-indexed list

                        try:
                            if value != "None":
                                # For mode values, we need to handle non-numeric responses
                                if answer_type == 'mode':
                                    try:
                                        value = float(value)
                                    except (ValueError, TypeError):
                                        # Skip non-numeric mode values
                                        continue
                                else:
                                    value = float(value)

                                year_data[question] = value
                                has_question_data = True
                        except (ValueError, TypeError):
                            pass

                if has_question_data:
                    country_data.append(year_data)

            # Convert to DataFrame
            if len(country_data) >= 3:  # Need at least 3 data points for meaningful correlation
                country_has_sufficient_data = True
                df = pd.DataFrame(country_data)

                # Calculate correlation between trade quantity and each question
                for question in questions:
                    # Use full question name without truncation
                    if question in df.columns:
                        # Check if we have enough non-NaN values
                        valid_data = df[['Trade_Quantity', question]].dropna()

                        if len(valid_data) >= 3:
                            # Check for constant values (no variation)
                            if valid_data['Trade_Quantity'].std() == 0 or valid_data[question].std() == 0:
                                # Can't calculate correlation when one variable is constant
                                continue

                            try:
                                # Use numpy's corrcoef which is more robust
                                corr_matrix = np.corrcoef(valid_data['Trade_Quantity'], valid_data[question])
                                corr = corr_matrix[0, 1]  # Get the correlation coefficient

                                # Check if correlation is valid (not NaN or infinite)
                                if pd.notna(corr) and np.isfinite(corr):
                                    if country_name not in country_question_correlations:
                                        country_question_correlations[country_name] = {}

                                    country_question_correlations[country_name][question] = corr

                                    # Track valid questions
                                    if question not in valid_questions:
                                        valid_questions.append(question)
                            except Exception as e:
                                print(f"Correlation calculation error for {country_name}, {question}: {e}")
                                continue

            if country_has_sufficient_data and country_name in country_question_correlations:
                valid_countries.append(country_name)

        if not valid_countries or not valid_questions:
            # Return empty figure with a message
            fig = go.Figure()
            fig.update_layout(
                title="Insufficient time series data for correlation analysis",
                template='plotly_white'
            )

            info_content = html.Div([
                html.H4("Insufficient Data"),
                html.P("Not enough years with both ESS and trade data to calculate meaningful correlations for any country.")
            ])

            return fig, info_content

        # Create correlation matrix for heatmap - transpose to switch the axes
        correlation_matrix = []

        # Sort countries and questions for better visualisation
        valid_countries.sort()
        valid_questions.sort()

        # Create z values for heatmap - build by question first (rows) instead of by country
        for question in valid_questions:
            question_row = []
            for country in valid_countries:
                if country in country_question_correlations and question in country_question_correlations[country]:
                    question_row.append(country_question_correlations[country][question])
                else:
                    question_row.append(None)  # No correlation data
            correlation_matrix.append(question_row)

        # Create the heatmap with switched axes
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            y=valid_questions,  # Questions on y-axis
            x=valid_countries,  # Countries on x-axis
            colorscale='RdBu_r',  # Red for negative correlations, blue for positive
            zmid=0,  # Center the color scale at 0
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title=dict(
                    text="Correlation",
                    side="right"
                )
            ),
            hovertemplate='Country: %{x}<br>Question: %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))

        # Update layout with switched axes
        fig.update_layout(
            title=f"Country-Level Correlations: Arms {trade_action} vs {display_name} {answer_type_text}",
            xaxis_title="Countries",
            yaxis_title="ESS Questions",
            height=800,
            margin=dict(l=300, r=50, t=80, b=80),  # More space for question labels on y-axis
            template='plotly_white',
            xaxis=dict(tickangle=-45)  # Angle country labels for better readability
        )

        # Prepare information panel content
        info_content = html.Div([
            html.H4(f"Country-Level Correlation Analysis: {display_name} ({answer_type_text}) vs. Arms {trade_action}"),
            html.P(f"This heatmap shows the correlation between the quantity of arms {trade_action.lower()} and {answer_type_text.lower()} to ESS questions for each country over time."),
            html.P("Interpretation:"),
            html.Ul([
                html.Li("Values closer to +1 or -1 indicate stronger correlations"),
                html.Li("White or missing cells indicate no correlation or insufficient data")
            ]),
            html.P(f"Displaying {len(valid_countries)} countries and {len(valid_questions)} questions with sufficient time series data for correlation analysis.")
        ])

        return fig, info_content

    except Exception as e:
        # Return error figure
        return go.Figure(layout={
            'title': f"Error: {str(e)}",
            'template': 'plotly_white'
        }), html.P(f"Error calculating correlation matrix: {str(e)}")

# Callback to switch between different visualisations
@callback(
    Output('page-content', 'children'),
    [Input('url', 'hash')]
)
def display_page(hash_value):
    if hash_value == '#trade-quantity':
        return trade_quantity_layout
    elif hash_value == '#correlation-matrix':
        return correlation_matrix_layout
    else:  # Default to ESS
        return ess_layout

# Callback to update active link in navbar
@callback(
    [Output('ess-link', 'active'),
     Output('trade-quantity-link', 'active'),
     Output('correlation-matrix-link', 'active')],
    [Input('url', 'hash')]
)
def update_active_link(hash_value):
    if hash_value == '#trade-quantity':
        return False, True, False
    elif hash_value == '#correlation-matrix':
        return False, False, True
    else:  # Default to ESS
        return True, False, False

def main():
    print("Starting Unified Data Visualisation Dashboard...")
    app.run_server(debug=True)

if __name__ == "__main__":
    main()
