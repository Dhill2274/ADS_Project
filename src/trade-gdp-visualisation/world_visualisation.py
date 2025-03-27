import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

# Load your government spending data
# Assuming format: country, year, spending
def load_data(filepath):
    """Load and prepare the government spending data."""
    df = pd.read_csv(filepath)
    
    # Ensure data is properly formatted
    df['Year'] = df['Year'].astype(int)
    df['Country'] = df['Country'].astype(str)
    df['Spending'] = df['Spending'].astype(float)
    
    # Calculate year-over-year percentage change
    df = df.sort_values(['Country', 'Year'])
    df['Change'] = df.groupby('Country')['Spending'].pct_change() * 100
    
    return df

# Load world shapefile data using geopandas
def load_world_data():
    """Load world map data."""
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Rename columns to match your data if needed
    world = world.rename(columns={'name': 'Country'})
    return world

def create_animation(spending_data, world_data, output_filename='government_spending_animation.mp4'):
    """Create animation of world map with countries resizing based on spending changes."""
    # Get unique years
    years = sorted(spending_data['Year'].unique())
    
    # Setup the figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Set up colormap for positive/negative changes
    cmap = plt.cm.RdYlGn  # Red (decrease) to Green (increase)
    norm = Normalize(vmin=-20, vmax=20)  # Adjust range as needed
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.02)
    cbar.set_label('Year-over-Year Change in Government Spending (%)')
    
    # Create title and annotation for the year
    title = ax.set_title('Government Spending Changes: 1950-2022', fontsize=16)
    year_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, 
                       ha='right', va='top', fontsize=14)
    
    # Function to create each frame for animation
    def update(year_idx):
        year = years[year_idx]
        ax.clear()
        
        # Get data for the current year
        current_year_data = spending_data[spending_data['Year'] == year]
        
        # Merge with world data
        merged = world_data.merge(current_year_data, on='Country', how='left')
        
        # Fill NaN values
        merged['Change'] = merged['Change'].fillna(0)
        
        # Calculate scaling factor for each country
        # Base size is 1, with changes amplified by scaling_factor
        scaling_factor = 0.05  # Adjust as needed
        merged['scale'] = 1 + (merged['Change'] * scaling_factor / 100)
        
        # Limit extreme scaling to prevent visual distortion
        merged['scale'] = merged['scale'].clip(0.5, 1.5)
        
        # Plot each country
        for idx, row in merged.iterrows():
            country = row['geometry']
            if pd.isna(row['Change']):
                color = 'lightgray'
            else:
                color = cmap(norm(row['Change']))
                
            # Scale the country based on spending change
            if not pd.isna(row['scale']):
                scale = row['scale']
                # Scale the geometry around its centroid
                centroid = country.centroid
                scaled_country = country.scale(scale, scale, origin=centroid)
                ax.add_patch(mpatches.Polygon(np.array(scaled_country.exterior.coords), 
                                             facecolor=color, edgecolor='black', linewidth=0.5))
            else:
                ax.add_patch(mpatches.Polygon(np.array(country.exterior.coords), 
                                             facecolor='lightgray', edgecolor='black', linewidth=0.5))
        
        # Update year text and set boundaries
        year_text.set_text(f'Year: {year}')
        ax.set_xlim(world_data.total_bounds[0], world_data.total_bounds[2])
        ax.set_ylim(world_data.total_bounds[1], world_data.total_bounds[3])
        ax.set_aspect('equal')
        ax.axis('off')
        
        return ax

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(years), blit=False)
    
    # Save as MP4
    anim.save(output_filename, writer='ffmpeg', fps=2, dpi=150)
    
    plt.close()
    return output_filename

def main():
    # Example usage
    spending_data = load_data('SIPRI-Milex-data-1948-2023.xlsx')
    world_data = load_world_data()
    
    # Ensure your data has proper country names that match the world map data
    # You might need to map country names between datasets
    animation_file = create_animation(spending_data, world_data)
    print(f"Animation saved as {animation_file}")
    
if __name__ == "__main__":
    main()