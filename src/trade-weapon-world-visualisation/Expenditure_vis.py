import pandas as pd
import plotly.graph_objects as go
import json
import numpy as np
from plotly.subplots import make_subplots
import os

# Define the colorscale globally for consistency
DETAILED_COLORSCALE = [
    [0.00, 'rgb(240,248,255)'], [0.01, 'rgb(225,238,250)'],
    [0.02, 'rgb(210,230,245)'], [0.04, 'rgb(190,220,240)'],
    [0.08, 'rgb(170,210,235)'], [0.12, 'rgb(150,200,230)'],
    [0.16, 'rgb(130,190,225)'], [0.20, 'rgb(110,180,220)'],
    [0.25, 'rgb(90,170,215)'],  [0.30, 'rgb(70,160,210)'],
    [0.35, 'rgb(60,150,205)'],  [0.40, 'rgb(50,140,200)'],
    [0.45, 'rgb(40,130,195)'],  [0.50, 'rgb(30,120,190)'],
    [0.60, 'rgb(20,110,185)'],  [0.70, 'rgb(15,100,180)'],
    [0.80, 'rgb(10,90,175)'],   [0.90, 'rgb(5,80,170)'],
    [0.95, 'rgb(0,70,160)'],    [0.98, 'rgb(0,50,140)'],
    [1.00, 'rgb(0,30,100)']
]

def load_data():
    """
    Load the military expenditure data and country coordinates
    """
    # Path to your military expenditure data - replace with your actual path
    path = "datasets/arms/gdp-expenditure.csv"
    # Treat '...' and 'xxx' as NaN values during loading
    df = pd.read_csv(path, na_values=['...', 'xxx'])

    # Assuming first column is country names and rest are years
    # Make sure data is in wide format with years as columns

    # Load the same country coordinates file used by the trade paths visualization
    with open('src/trade-weapon-world-visualisation/country_coordinates.json', 'r') as f:
        country_coords = json.load(f)

    return df, country_coords

def preprocess_data(df):
    """
    Process the military expenditure data for visualization
    """
    # Get all years from column names (assuming first column is Country)
    years = [col for col in df.columns if col != 'Country']
    years.sort()

    # Create a dictionary to store data for each year
    year_data = {}

    for year in years:
        # Get data for this year
        year_df = df[['Country', year]].copy()
        year_df.columns = ['Country', 'Expenditure']

        # Ensure 'Expenditure' is treated as string before replacing '%'
        if year_df['Expenditure'].dtype == 'object':
             year_df['Expenditure'] = year_df['Expenditure'].astype(str).str.replace('%', '', regex=False)

        # Convert Expenditure to numeric, coercing errors (like remaining strings or previous NaNs)
        year_df['Expenditure'] = pd.to_numeric(year_df['Expenditure'], errors='coerce')

        # Remove NaN/NaT values AFTER conversion
        year_df = year_df.dropna(subset=['Expenditure'])

        # Store in the dictionary
        year_data[year] = year_df

    return years, year_data

def create_choropleth_frames(years, year_data):
    """
    Create frames for the choropleth animation with auto-adjusting color scale
    """
    frames = []
    for year in years:
        frame_data = []
        current_year_df = year_data[year]

        # Skip if no data for this year
        if current_year_df.empty:
             continue

        # Calculate max value for the current year
        current_max_value = current_year_df['Expenditure'].max()
        # Ensure max_value is slightly > 0 if min is 0 to avoid zmin=zmax
        if pd.isna(current_max_value) or current_max_value <= 0:
            current_max_value = 0.01 # Set a small default max if data is zero or missing

        # Create choropleth trace for this year
        choropleth = go.Choropleth(
            locations=current_year_df['Country'],
            z=current_year_df['Expenditure'], # Numeric data
            locationmode='country names',
            colorscale=DETAILED_COLORSCALE, # Consistent colorscale
            zmin=0,
            zmax=current_max_value, # Auto-adjust zmax for this year
            hovertemplate='%{location}<br>Military Expenditure: %{z:.2f}%<extra></extra>', # Format hover
            colorbar=dict(
                title=dict(
                    text='Military Expenditure (% of GDP)',
                    side='right', # Place title text on the right
                    font=dict(size=10) # Optional: adjust font size
                ),
                # titleangle is not a direct property, rotation is usually handled by layout/axis settings
                # We set side='right' here. Angle adjustment might need layout updates if possible for colorbar title.
            )
        )

        frame_data.append(choropleth)
        frames.append(go.Frame(data=frame_data, name=str(year)))

    return frames

def initialize_figure(year_data, years):
    """
    Initialize the figure with the first year's data, auto-adjusting color scale
    """
    fig = go.Figure()

    # Add initial choropleth for the first year
    first_year = years[0]
    first_year_df = year_data[first_year]

    # Handle case where first year might have no data
    if not first_year_df.empty:
        first_year_max = first_year_df['Expenditure'].max()
        if pd.isna(first_year_max) or first_year_max <= 0:
            first_year_max = 0.01 # Small default max

        fig.add_trace(
            go.Choropleth(
                locations=first_year_df['Country'],
                z=first_year_df['Expenditure'], # Numeric data
                locationmode='country names',
                colorscale=DETAILED_COLORSCALE, # Consistent colorscale
                zmin=0,
                zmax=first_year_max, # Auto-adjust zmax for the first year
                hovertemplate='%{location}<br>Military Expenditure: %{z:.2f}%<extra></extra>', # Format hover
                colorbar=dict(
                    title=dict(
                        text='Military Expenditure (% of GDP)',
                        side='right', # Place title text on the right
                        font=dict(size=10) # Optional: adjust font size
                    ),
                    # titleangle is not a direct property, rotation is usually handled by layout/axis settings
                    # We set side='right' here. Angle adjustment might need layout updates if possible for colorbar title.
                )
            )
        )
    # else: figure remains empty initially, will be populated by frames/animation


    return fig

def create_slider_and_controls(years):
    """
    Create the year slider and play/pause controls
    """
    # Create steps for the slider
    steps = []
    for year in years:
        step = dict(
            method="animate",
            args=[
                [str(year)],
                {"frame": {"duration": 300, "redraw": True},
                 "mode": "immediate",
                 "transition": {"duration": 300}}
            ],
            label=str(year)
        )
        steps.append(step)

    # Create slider
    sliders = [dict(
        active=0,
        yanchor="top",
        xanchor="left",
        currentvalue=dict(
            font=dict(size=16),
            prefix="Year: ",
            visible=True,
            xanchor="right"
        ),
        transition=dict(duration=300, easing="cubic-in-out"),
        pad=dict(b=10, t=50),
        len=0.9,
        x=0.1,
        y=0,
        steps=steps
    )]

    # Create play/pause buttons
    updatemenus = [dict(
        type="buttons",
        buttons=[
            dict(
                label="Play",
                method="animate",
                args=[None, {"frame": {"duration": 300, "redraw": True},
                             "fromcurrent": True,
                             "transition": {"duration": 300, "easing": "quadratic-in-out"}}]
            ),
            dict(
                label="Pause",
                method="animate",
                args=[[None], {"frame": {"duration": 0, "redraw": False},
                               "mode": "immediate",
                               "transition": {"duration": 0}}]
            )
        ],
        direction="left",
        pad=dict(r=10, t=70),
        showactive=False,
        x=0.1,
        xanchor="right",
        y=0,
        yanchor="top"
    )]

    return sliders, updatemenus

def military_expenditure_visualization():
    """
    Main function to create the military expenditure visualization
    """
    print("Loading data...")
    df, country_coords = load_data()

    print("Processing data...")
    years, year_data = preprocess_data(df)

    print("Initializing figure...")
    # Initialize figure (now auto-adjusts zmax internally)
    fig = initialize_figure(year_data, years)

    print("Creating frames...")
    # Create frames (now auto-adjusts zmax internally)
    frames = create_choropleth_frames(years, year_data)
    fig.frames = frames

    print("Setting up controls...")
    sliders, updatemenus = create_slider_and_controls(years)

    print("Finalizing visualization...")
    fig.update_layout(
        title_text='Military Expenditure by Country (% of GDP)',
        geo=dict(
            scope='world',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            projection=dict(type='miller'),
            showcountries=True,
            countrywidth=0.5,
        ),
        sliders=sliders,
        updatemenus=updatemenus,
        height=800
    )

    return fig

def main():
    # This function will create both visualizations on the same page
    print("Creating military expenditure visualization...")
    military_fig = military_expenditure_visualization()

    # Import the trade paths visualization function from the original code
    # Assuming the original module is named trade_paths_visualization.py
    import importlib.util

    # Check if we need to run the trade paths visualization
    run_trade_paths = True
    try:
        # Try to import the original module
        spec = importlib.util.spec_from_file_location("trade_paths", "/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Desktop/University Engmaths/Applied data science/ADS Coursework/Repo/ADS_Project/src/trade-weapon-world-visualisation/trade_vis.py")
        trade_paths = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trade_paths)

        # Access the main function that creates the figure
        trade_fig = trade_paths.main(return_fig=True)  # Assuming the main function can return the figure
    except Exception as e:
        print(f"Could not import trade paths visualization: {e}")
        run_trade_paths = False

    # Create a subplot with both visualizations
    if run_trade_paths:
        combined_fig = make_subplots(
            rows=2, cols=1,
            specs=[[{"type": "choropleth"}], [{"type": "scattergeo"}]],
            subplot_titles=("Military Expenditure by Country", "Trade Paths by Year"),
            vertical_spacing=0.1
        )

        # Add military expenditure visualization to the first row
        for trace in military_fig.data:
            combined_fig.add_trace(trace, row=1, col=1)

        # Add trade paths visualization to the second row
        for trace in trade_fig.data:
            combined_fig.add_trace(trace, row=2, col=1)

        # Update layout
        combined_fig.update_layout(
            height=1600,
            title_text="Military Expenditure and Trade Path Visualizations"
        )

        # Handle frames from both visualizations
        # This requires more complex frame handling that merges both sets of frames

        print("Rendering combined visualization...")
        combined_fig.show()
    else:
        # Just show the military expenditure visualization
        print("Rendering military expenditure visualization only...")
        military_fig.show()

# Modified version of the main function from the original code
# This allows us to use the trade paths visualization as a component
def create_combined_visualizations():
    """
    Function to create both visualizations and display them on one page
    """
    # Create the standalone military expenditure visualization
    military_fig = military_expenditure_visualization()

    # Allow the user to switch between visualizations
    military_fig.show()

    # Run the original trade paths visualization code
    print("\nPress any key to view the Trade Paths visualization...")
    input()

    # Run the original main function which will show the trade paths visualization
    # This assumes the original main file is in the same directory
    exec(open("original_trade_paths_file.py").read())

if __name__ == "__main__":
    # Run the main function to create the visualization
    main()