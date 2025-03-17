import pandas as pd
import plotly.graph_objects as go
import json
import math
import plotly.express as px
import concurrent.futures
from tqdm import tqdm
import multiprocessing
import pyproj

NUM_STEPS_PER_PATH = 20

def load_data():
    path = "Datasets/trades-weighted.csv"
    df = pd.read_csv(path)
    df = df[df['Quantity'] > 0]

    with open('src/trade-visualisation/country_coordinates.json', 'r') as f:
        country_coords = json.load(f)

    return df, country_coords

def get_great_circle_points(start_coords, end_coords, num_points=20):
    if start_coords == end_coords:
        return [start_coords[0]] * num_points, [start_coords[1]] * num_points

    g = pyproj.Geod(ellps='WGS84')
    lon_start, lat_start = start_coords[1], start_coords[0]
    lon_end, lat_end = end_coords[1], end_coords[0]

    if num_points > 2:
        points = g.npts(lon_start, lat_start, lon_end, lat_end, num_points-2)
        lons = [lon_start] + [p[0] for p in points] + [lon_end]
        lats = [lat_start] + [p[1] for p in points] + [lat_end]
    else:
        lons = [lon_start, lon_end]
        lats = [lat_start, lat_end]

    return lats, lons

def preprocess_trade_data(df, country_coords):
    print("Pre-processing data...")

    years = [int(year) for year in df['Year of order'].unique() if not pd.isna(year)]
    years.sort()

    year_paths = {year: [] for year in years}

    for index, row in df.iterrows():
        from_country = row['From']
        to_country = row['To']
        quantity = row['Quantity']
        weight = row['Weight']
        weapon_desc = row['Weapon description']
        start_year = row['Year of order']

        if pd.isna(start_year):
            continue

        duration = row.get('Duration', 1)
        if pd.isna(duration) or duration <= 0:
            duration = 1

        start_year = int(start_year)
        duration = int(duration)

        if from_country in country_coords and to_country in country_coords:
            from_coords = country_coords[from_country]
            to_coords = country_coords[to_country]

            path_lats, path_lons = get_great_circle_points(from_coords, to_coords, 50)

            for year_index, active_year in enumerate(range(start_year, start_year + duration)):
                if active_year in year_paths:
                    is_new_order = (active_year == start_year)
                    trade_id = f"{from_country}-{to_country}-{start_year}-{index}"

                    year_paths[active_year].append((
                        from_coords, to_coords, quantity, from_country, to_country,
                        weight, weapon_desc, is_new_order, trade_id, path_lats, path_lons,
                        duration, year_index
                    ))

    return years, year_paths

def create_color_mapping(year_paths, years):
    all_weights = []
    for year in years:
        for path in year_paths[year]:
            all_weights.append(path[5])

    min_weight = min(all_weights) if all_weights else 0
    max_weight = max(all_weights) if all_weights else 1
    unique_weights = sorted(list(set(all_weights)))

    colorscale = px.colors.sequential.Sunset
    weight_colors = {}

    for weight in unique_weights:
        norm_weight = (weight - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 0.5
        color_idx = int(norm_weight * (len(colorscale) - 1))
        color_idx = max(0, min(color_idx, len(colorscale) - 1))
        weight_colors[weight] = colorscale[color_idx]

    return min_weight, max_weight, unique_weights, weight_colors

def create_frame(frame_data):
    year, step, year_paths, max_paths, min_weight, max_weight, unique_weights, weight_colors = frame_data
    frame_traces = []

    frame_traces.append(
        go.Scattergeo(
            lon=[None],
            lat=[None],
            mode='markers',
            marker=dict(
                size=0,
                colorscale='sunset',
                colorbar=dict(
                    title="Severity (Weight)",
                    tickmode="array",
                    tickvals=unique_weights,
                    ticktext=unique_weights,
                    ticks="outside"
                ),
                cmin=min_weight,
                cmax=max_weight
            ),
            hoverinfo='none',
            showlegend=False
        )
    )

    for i in range(max_paths):
        if i < len(year_paths[year]):
            path = year_paths[year][i]
            from_coords, to_coords, quantity, from_country, to_country, weight, weapon_desc, is_new_order, trade_id, path_lats, path_lons, duration, year_index = path

            width = 0.5 + 2 * math.log(quantity + 1, 10) if quantity > 0 else 0.5
            width = min(width, 10)

            color = weight_colors[weight]
            opacity = 0.5
            order_status = "New Order" if is_new_order else "Ongoing Order"

            frame_traces.append(
                go.Scattergeo(
                    lon=[from_coords[1], to_coords[1]],
                    lat=[from_coords[0], to_coords[0]],
                    mode='lines',
                    line=dict(
                        width=width,
                        color=color
                    ),
                    opacity=opacity,
                    hoverinfo='text',
                    hovertext=f"From: {from_country}<br>To: {to_country}<br>Quantity: {quantity}<br>Weight: {weight}<br>Weapon: {weapon_desc}<br>Status: {order_status}",
                    showlegend=False,
                )
            )
        else:
            frame_traces.append(
                go.Scattergeo(
                    lon=[None, None],
                    lat=[None, None],
                    mode='lines',
                    line=dict(width=0),
                    opacity=0,
                    hoverinfo='none',
                    showlegend=False,
                )
            )

    for i in range(max_paths):
        if i < len(year_paths[year]):
            path = year_paths[year][i]
            from_coords, to_coords, quantity, from_country, to_country, weight, weapon_desc, is_new_order, trade_id, path_lats, path_lons, duration, year_index = path

            color = weight_colors[weight]

            total_points = len(path_lats)
            segment_start = int((year_index / duration) * total_points)
            segment_end = int(((year_index + 1) / duration) * total_points)

            segment_length = segment_end - segment_start
            progress = step / NUM_STEPS_PER_PATH
            current_point_index = segment_start + int(progress * segment_length)
            current_point_index = min(current_point_index, total_points - 1)

            dot_lat = path_lats[current_point_index]
            dot_lon = path_lons[current_point_index]

            year_progress = f"Year {year_index+1} of {duration}"
            if duration > 1:
                segment_percent = int((year_index / duration) * 100)
                segment_end_percent = int(((year_index + 1) / duration) * 100)
                progress_display = f"{segment_percent}% to {segment_end_percent}%"
            else:
                progress_display = "0% to 100%"

            frame_traces.append(
                go.Scattergeo(
                    lon=[dot_lon],
                    lat=[dot_lat],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=color,
                        symbol='circle',
                        line=dict(
                            width=1,
                            color='black'
                        )
                    ),
                    opacity=1,
                    hoverinfo='text',
                    hovertext=f"From: {from_country}<br>To: {to_country}<br>{year_progress}<br>Progress: {progress_display}",
                    showlegend=False,
                )
            )
        else:
            frame_traces.append(
                go.Scattergeo(
                    lon=[None],
                    lat=[None],
                    mode='markers',
                    marker=dict(size=0),
                    opacity=0,
                    hoverinfo='none',
                    showlegend=False,
                )
            )

    frame_name = f"{year}_{step}"
    return frame_name, frame_traces

def initialize_figure(min_weight, max_weight, unique_weights, max_paths, year_paths, years):
    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lon=[None],
            lat=[None],
            mode='markers',
            marker=dict(
                size=0,
                colorscale='sunset',
                colorbar=dict(
                    title="Severity (Weight)",
                    tickmode="array",
                    tickvals=unique_weights,
                    ticktext=unique_weights,
                    ticks="outside"
                ),
                cmin=min_weight,
                cmax=max_weight
            ),
            hoverinfo='none',
            showlegend=False
        )
    )

    for i in range(max_paths):
        fig.add_trace(
            go.Scattergeo(
                lon=[None, None],
                lat=[None, None],
                mode='lines',
                line=dict(width=0),
                opacity=0,
                hoverinfo='none',
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scattergeo(
                lon=[None],
                lat=[None],
                mode='markers',
                marker=dict(
                    size=8,
                    color='white',
                    symbol='circle',
                    line=dict(
                        width=1,
                        color='black'
                    )
                ),
                opacity=0,
                hoverinfo='none',
                showlegend=False,
            )
        )

    return fig

def process_frames_in_parallel(years, year_paths, max_paths, min_weight, max_weight, unique_weights, weight_colors):
    frame_data_list = []
    for year in years:
        for step in range(NUM_STEPS_PER_PATH):
            frame_data_list.append((year, step, year_paths, max_paths, min_weight, max_weight, unique_weights, weight_colors))

    print(f"Creating {len(frame_data_list)} frames using parallel processing...")
    frames = []
    frame_dict = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(
            executor.map(create_frame, frame_data_list),
            total=len(frame_data_list),
            desc="Processing frames"
        ))

        for frame_name, frame_traces in results:
            frame_dict[frame_name] = frame_traces

    for year in years:
        for step in range(NUM_STEPS_PER_PATH):
            frame_name = f"{year}_{step}"
            if frame_name in frame_dict:
                frames.append(go.Frame(data=frame_dict[frame_name], name=frame_name))

    return frames

def update_initial_data(fig, year_paths, max_paths, weight_colors):
    first_year = next(iter(year_paths))
    paths = year_paths[first_year]

    for i in range(max_paths):
        if i < len(paths):
            path = paths[i]
            from_coords, to_coords, quantity, from_country, to_country, weight, weapon_desc, is_new_order, trade_id, path_lats, path_lons, duration, year_index = path

            width = 0.5 + 2 * math.log(quantity + 1, 10) if quantity > 0 else 0.5
            width = min(width, 10)

            color = weight_colors[weight]
            opacity = 0.5
            order_status = "New Order" if is_new_order else "Ongoing Order"

            fig.data[i*2+1].update(
                lon=[from_coords[1], to_coords[1]],
                lat=[from_coords[0], to_coords[0]],
                mode='lines',
                line=dict(
                    width=width,
                    color=color
                ),
                opacity=opacity,
                hoverinfo='text',
                hovertext=f"From: {from_country}<br>To: {to_country}<br>Quantity: {quantity}<br>Weight: {weight}<br>Weapon: {weapon_desc}<br>Status: {order_status}",
            )

            segment_start = int((year_index / duration) * len(path_lats))

            fig.data[i*2+2].update(
                lon=[path_lons[segment_start]],
                lat=[path_lats[segment_start]],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    symbol='circle',
                    line=dict(
                        width=1,
                        color='black'
                    )
                ),
                opacity=1,
                hoverinfo='text',
                hovertext=f"From: {from_country}<br>To: {to_country}",
            )

def create_slider_and_controls(years):
    sliders = [{
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 16},
            'prefix': 'Year: ',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 30, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }]

    print("Setting up slider...")
    for year in years:
        frame_name = f"{year}_0"
        slider_step = {
            'args': [
                [frame_name],
                {'frame': {'duration': 50, 'redraw': True},
                 'mode': 'immediate',
                 'transition': {'duration': 30}}
            ],
            'label': str(year),
            'method': 'animate'
        }
        sliders[0]['steps'].append(slider_step)

    updatemenus = [{
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 30, 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': 20, 'easing': 'quadratic-in-out'}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                 'mode': 'immediate',
                                 'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 70},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }]

    return sliders, updatemenus

def main():
    df, country_coords = load_data()
    years, year_paths = preprocess_trade_data(df, country_coords)
    min_weight, max_weight, unique_weights, weight_colors = create_color_mapping(year_paths, years)

    max_paths = max([len(year_paths[year]) for year in years])

    fig = initialize_figure(min_weight, max_weight, unique_weights, max_paths, year_paths, years)

    frames = process_frames_in_parallel(years, year_paths, max_paths, min_weight, max_weight, unique_weights, weight_colors)

    update_initial_data(fig, year_paths, max_paths, weight_colors)

    print("Setting frames...")
    fig.frames = frames

    sliders, updatemenus = create_slider_and_controls(years)

    fig.update_layout(
        title='Trade Paths by Year',
        geo=dict(
            scope='world',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(0, 0, 0)',
            projection=dict(type='miller'),
            showcountries=True,
            countrywidth=0.5,
        ),
        sliders=sliders,
        updatemenus=updatemenus
    )

    print("Rendering visualisation...")
    fig.show()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()