from dataset import Dataset, DATASET_DISPLAY_NAMES
import plotly.graph_objects as go
from dash import Dash, dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc

countries = {
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

datasets = ["chars", "media", "politics", "socio", "values", "wellbeing"]

# Create Dash app with Bootstrap theme
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="ESS Data Visualisation"  # This sets the browser tab title
)

# App layout with improved Bootstrap components
app.layout = dbc.Container([
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
                                id='dataset-dropdown',
                                options=[{'label': DATASET_DISPLAY_NAMES.get(dataset, dataset.capitalize()),
                                          'value': dataset}
                                         for dataset in datasets],
                                value=datasets[0],
                                clearable=False,
                                className="mb-3"
                            ),
                        ], xs=12, md=6),

                        dbc.Col([
                            html.Label('Select Question:', className="fw-bold"),
                            dcc.Dropdown(
                                id='question-dropdown',
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
                                id='visualization-type',
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
                id="loading-graph",
                type="circle",
                children=[
                    dcc.Graph(
                        id='survey-graph',
                        style={'height': '600px'},
                        figure=go.Figure(layout={
                            'title': 'Select a dataset and question to view data',
                            'xaxis': {'title': 'ESS Round', 'range': [0.5, 11.5], 'dtick': 1},
                            'yaxis': {'title': 'Response Value', 'range': [1, None]},
                            'template': 'plotly_white'  # Add a cleaner template
                        })
                    )
                ]
            )
        ])
    ], className="mb-4 shadow-sm"),

    dbc.Card([
        dbc.CardHeader("Dataset Information"),
        dbc.CardBody(id='dataset-info')
    ], className="shadow-sm"),

    html.Footer([
        html.P("European Social Survey Data Visualisation Tool", className="text-center text-muted mt-4")
    ])
], fluid=True, className="px-4 py-3")

# Callback to update question dropdown when dataset changes
@callback(
    [Output('question-dropdown', 'options'),
     Output('question-dropdown', 'value'),
     Output('dataset-info', 'children')],
    Input('dataset-dropdown', 'value')
)
def update_question_dropdown(selected_dataset):
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

# Update the callback to show progress for question loading
@callback(
    Output('survey-graph', 'figure'),
    [Input('dataset-dropdown', 'value'),
     Input('question-dropdown', 'value'),
     Input('visualization-type', 'value')]
)
def update_graph(selected_dataset, selected_question, viz_type):
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
        for country_code, country_name in countries.items():
            if country_code in dataset_obj.countryLabels:
                y_values = question_data[country_code]

                # Convert answers to numerical values if possible for better visualization
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

def main():
    # Run the Dash app
    app.run_server(debug=True)

if __name__ == "__main__":
    main()
