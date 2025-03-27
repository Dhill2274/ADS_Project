import pandas as pd
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
import time
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut
import json

def get_country_coordinates(country, max_retries=3, retry_delay=2):
    print(f"Getting coordinates for {country}...")
    geolocator = Nominatim(user_agent="ADS_Project_Geocoder")
    retries = 0
    while retries < max_retries:
        try:
            location = geolocator.geocode(country, timeout=10) # Added timeout
            print(f"Found coordinates for {country}")
            if location:
                return (location.latitude, location.longitude)
            else:
                print(f"Could not find coordinates for {country}")
                return None
        except (GeocoderUnavailable, GeocoderTimedOut) as e:
            retries += 1
            print(f"Geocoding failed for {country} (attempt {retries}/{max_retries}): {e}")
            if retries < max_retries:
                time.sleep(retry_delay) # Wait before retrying
        except Exception as e:
            print(f"An unexpected error occurred for {country}: {e}")
            return None
    print(f"Max retries reached for {country}. Geocoding failed.")
    return None

def get_all_country_coordinates(countries):
    coordinates = {}
    for country in countries:
        coords = get_country_coordinates(country)
        if coords:
            coordinates[country] = coords
        else:
            print(f"Could not find coordinates for {country}")
    return coordinates

def main():
    path = "Datasets/trades-weighted.csv"
    df = pd.read_csv(path)

    from_countries = list(df['From'].unique())
    to_countries = list(df['To'].unique())

    countries = from_countries + to_countries
    countries = [country for country in countries if country not in ["unknown recipient(s)", "unknown supplier(s)", "unknown"]]
    countries = [str(country) for country in countries if "*" not in str(country) and "**" not in str(country)]
    countries = set(countries)

    coords = get_all_country_coordinates(countries)

    country_coords = {}
    for country in countries:
        if country in coords:
            country_coords[country] = coords[country]
        else:
            print(f"Could not find coordinates for {country}")
            country_coords[country] = (0, 0)

    with open('src/trade-visualisation/country_coordinates.json', 'w') as f:
        json.dump(country_coords, f)

if __name__ == "__main__":
    main()