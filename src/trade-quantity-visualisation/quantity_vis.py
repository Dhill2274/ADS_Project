import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import json
import plotly.express as px

# Define ESS countries - European countries included in the European Social Survey
ESS_COUNTRIES = [
    "Austria", "Belgium", "Bulgaria", "Switzerland", "Cyprus", "Czechia",
    "Germany", "Denmark", "Estonia", "Spain", "Finland", "France",
    "United Kingdom", "Hungary", "Ireland", "Israel", "Lithuania",
    "Netherlands", "Norway", "Poland", "Portugal", "Russian Federation",
    "Sweden", "Slovenia", "Slovakia", "Ukraine"
]

# Load and prepare the data
def load_data():
    print("Loading data...")
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

# Create Dash app
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Arms Trade Quantity Visualisation"
)

# App layout
app.layout = dbc.Container([
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
                                id='view-type',
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
                                id='country-filter',
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
                id="loading-graph",
                type="circle",
                children=[
                    dcc.Graph(
                        id='quantity-graph',
                        style={'height': '750px'},
                        figure=go.Figure(layout={
                            'title': 'Select parameters to view arms trade quantity data',
                            'xaxis': {
                                'title': 'Year',
                                'rangeslider': {'visible': True},
                                'range': [2002, 2023],
                                'dtick': 4,  # Show tick marks every 4 years instead of 2
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

# Callback to update the graph based on user selections
@callback(
    [Output('quantity-graph', 'figure'),
     Output('trade-info', 'children')],
    [Input('view-type', 'value'),
     Input('country-filter', 'value')]
)
def update_graph(view_type, country_filter):
    # Load and process data
    df, _ = load_data()
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
        countries_to_display = [country for country in countries_with_data if country in ESS_COUNTRIES]
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
        dtick=4,  # Show tick marks every 4 years instead of 2
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

def main():
    print("Starting Arms Trade Quantity Visualisation app...")
    app.run_server(debug=True)

if __name__ == "__main__":
    main()