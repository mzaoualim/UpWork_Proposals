import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# --- 1. CONFIGURATION AND DATA GENERATION ---

st.set_page_config(
    page_title="Australian Property Heatmap",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Coordinates for 5 major Australian localities (used for synthetic data generation)
LOCATIONS = {
    'Sydney': (-33.85, 151.21),
    'Melbourne': (-37.81, 144.96),
    'Brisbane': (-27.47, 153.02),
    'Perth': (-31.95, 115.86),
    'Adelaide': (-34.92, 138.59),
}
PROPERTY_TYPES = ['House', 'Unit', 'Townhouse']
YEARS = [2022, 2023, 2024]
BASE_PRICE = 800000

@st.cache_data
def generate_synthetic_data(n_points_per_locality=100):
    """Generates a synthetic DataFrame for property prices with Lat/Lon."""
    data = []
    
    for locality, (base_lat, base_lon) in LOCATIONS.items():
        for year in YEARS:
            for prop_type in PROPERTY_TYPES:
                # Determine price base and volatility
                price_multiplier = 1.0 + (year - 2022) * 0.08  # 8% annual growth
                if prop_type == 'Unit':
                    price_multiplier *= 0.7
                elif prop_type == 'Townhouse':
                    price_multiplier *= 0.85
                
                # Generate multiple points slightly offset from the center
                for _ in range(n_points_per_locality):
                    # Add small random offset to simulate different properties
                    lat_offset = np.random.uniform(-0.1, 0.1)
                    lon_offset = np.random.uniform(-0.1, 0.1)
                    
                    # Calculate median price with some noise
                    price_noise = np.random.normal(0, 50000)
                    median_price = int(BASE_PRICE * price_multiplier + price_noise)
                    median_price = max(100000, median_price) # Ensure price is reasonable
                    
                    data.append({
                        'Locality': locality,
                        'Year': year,
                        'PropertyType': prop_type,
                        'MedianPrice': median_price,
                        'Lat': base_lat + lat_offset,
                        'Lon': base_lon + lon_offset
                    })
    
    df = pd.DataFrame(data)
    # Ensure Lat/Lon columns are floats
    df['Lat'] = df['Lat'].astype(float)
    df['Lon'] = df['Lon'].astype(float)
    return df

# Load data
df = generate_synthetic_data()


# --- 2. STREAMLIT APP LAYOUT AND FILTERS ---

st.title("🏡 Australian Property Price Heatmap Analyzer")

# Sidebar for user filtering
with st.sidebar:
    st.header("Filter Options")

    # Filter 1: Year
    selected_year = st.slider(
        "Select Year",
        min_value=min(YEARS),
        max_value=max(YEARS),
        value=max(YEARS),
        step=1
    )

    # Filter 2: Locality Names
    all_localities = sorted(df['Locality'].unique())
    selected_localities = st.multiselect(
        "Select Localities",
        options=all_localities,
        default=all_localities,
        placeholder="Choose up to 5 localities"
    )

    # Filter 3: Property Type
    all_types = sorted(df['PropertyType'].unique())
    selected_types = st.multiselect(
        "Select Property Types",
        options=all_types,
        default=['House'],
        placeholder="Choose property types"
    )
    
    st.markdown("""---""")
    st.markdown("**Visualization Details**")
    st.info("The heatmap color intensity is weighted by the Median Price.")


# --- 3. FILTERING DATA AND PREPARING MAP DATA ---

if not selected_localities or not selected_types:
    st.error("Please select at least one Locality and one Property Type to generate the heatmap.")
else:
    # Apply filters
    filtered_df = df[
        (df['Year'] == selected_year) &
        (df['Locality'].isin(selected_localities)) &
        (df['PropertyType'].isin(selected_types))
    ].copy()

    # --- 4. MAP GENERATION LOGIC ---

    if filtered_df.empty:
        st.warning(f"No data available for the selected criteria in {selected_year}.")
    else:
        # Calculate the center of the filtered data for map centering
        center_lat = filtered_df['Lat'].mean()
        center_lon = filtered_df['Lon'].mean()
        
        # Create a Folium map object
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=6, 
            tiles="OpenStreetMap" # Using OpenStreetMap tiles
        )

        # Prepare data for HeatMap plugin: [[lat, lon, weight], ...]
        # We need to normalize the price to ensure the heatmap intensity is visible and useful.
        # Use min-max scaling for the weights (0 to 1)
        min_price = filtered_df['MedianPrice'].min()
        max_price = filtered_df['MedianPrice'].max()
        
        # Avoid division by zero if all prices are identical
        price_range = max_price - min_price
        
        if price_range > 0:
            filtered_df['Weight'] = (filtered_df['MedianPrice'] - min_price) / price_range
        else:
            filtered_df['Weight'] = 0.5 # Default weight if prices are all the same

        # Create the list of [Latitude, Longitude, Weight]
        heat_data = filtered_df[['Lat', 'Lon', 'Weight']].values.tolist()
        
        # Add the HeatMap layer to the map
        HeatMap(
            heat_data,
            name='Property Price Heatmap',
            radius=20,          # Adjust radius for better visual grouping
            blur=15,            # Adjust blur for smooth transition
            min_opacity=0.3,
            max_val=1.0         # Max intensity corresponds to max weight (max price)
        ).add_to(m)

        # Display the map in Streamlit
        st_folium(m, height=600, width="100%")
        
        # Display summary statistics
        st.subheader("Summary of Filtered Data")
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Median Price", f"${filtered_df['MedianPrice'].mean():,.0f}")
        col2.metric("Min Median Price", f"${min_price:,.0f}")
        col3.metric("Max Median Price", f"${max_price:,.0f}")
        
        st.dataframe(
            filtered_df[['Locality', 'PropertyType', 'Year', 'MedianPrice']].rename(
                columns={'MedianPrice': 'Median Price ($)'}
            ).sort_values(by='Median Price ($)', ascending=False).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
