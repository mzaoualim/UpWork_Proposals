import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium 
from folium.plugins import HeatMap
from folium.features import DivIcon # Needed for custom labels on the map
import colorsys

# --- 1. CONFIGURATION AND DATA GENERATION (Unchanged from previous version) ---

st.set_page_config(
    page_title="Australian Property Heatmap",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Coordinates for 5 major Australian localities (used for synthetic data generation)
# Note: I am using the general city coordinates as placeholders for the "locality centers"
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
                    # Spread the points slightly around the city center
                    lat_offset = np.random.uniform(-0.3, 0.3) 
                    lon_offset = np.random.uniform(-0.3, 0.3)
                    
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

# --- 2. UTILITY FUNCTIONS FOR CHOROPLETH-STYLE VISUALIZATION ---

def get_color_from_price(price, min_price, max_price):
    """Maps a price value to a blue color gradient (light to dark)."""
    if min_price == max_price:
        # If all prices are the same, use a neutral blue
        return '#0070c0' 
        
    # Normalize price to a 0-1 range
    normalized_price = (price - min_price) / (max_price - min_price)
    
    # We want higher prices to be DARKER blue.
    # Light: #ADD8E6 (RGB 173, 216, 230) -> Dark: #104E8B (RGB 16, 78, 139)
    r1, g1, b1 = (173, 216, 230)
    r2, g2, b2 = (16, 78, 139)
    
    # Interpolate (r, g, b) values. We want lower price to be light, higher price to be dark.
    r = int(r1 + normalized_price * (r2 - r1))
    g = int(g1 + normalized_price * (g2 - g1))
    b = int(b1 + normalized_price * (b2 - b1))
    
    return f'#{r:02x}{g:02x}{b:02x}'


# --- 3. STREAMLIT APP LAYOUT AND FILTERS ---

st.title("🏡 Australian Property Price Map (Choropleth Style)")

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
    st.markdown("**Visualization Style**")
    st.info("This map uses custom color-coded labels to simulate suburb-level data aggregation, similar to a Choropleth map.")


# --- 4. FILTERING DATA AND PREPARING MAP DATA ---

if not selected_localities or not selected_types:
    st.error("Please select at least one Locality and one Property Type to generate the map.")
else:
    # Apply filters
    filtered_df = df[
        (df['Year'] == selected_year) &
        (df['Locality'].isin(selected_localities)) &
        (df['PropertyType'].isin(selected_types))
    ].copy()

    # --- 5. MAP GENERATION LOGIC ---

    if filtered_df.empty:
        st.warning(f"No data available for the selected criteria in {selected_year}.")
    else:
        
        # Aggregate the data by Locality to get a single data point (mean price) for visualization
        grouped_df = filtered_df.groupby('Locality').agg(
            MeanPrice=('MedianPrice', 'mean'),
            Lat=('Lat', 'mean'), 
            Lon=('Lon', 'mean')
        ).reset_index()

        # Calculate map center based on grouped data
        center_lat = grouped_df['Lat'].mean()
        center_lon = grouped_df['Lon'].mean()
        
        # Determine min/max price for color scaling
        min_price = grouped_df['MeanPrice'].min()
        max_price = grouped_df['MeanPrice'].max()

        # Create a Folium map object
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=5, # Zoom out slightly to see all Australian cities
            tiles="CartoDB positron" # Clean, light tileset for better visibility
        )

        # Iterate through grouped data and add custom DivIcon markers
        for index, row in grouped_df.iterrows():
            price = row['MeanPrice']
            locality_name = row['Locality']
            
            # Get color based on price
            fill_color = get_color_from_price(price, min_price, max_price)
            text_color = '#FFFFFF' if price > (min_price + max_price) / 2 else '#000000' # Simple rule for contrast
            
            # Format price for display
            formatted_price = f"${price:,.0f}"
            
            # --- Custom HTML/CSS for the Label (DivIcon) ---
            html = f"""
                <div style="
                    background-color: {fill_color};
                    color: {text_color};
                    border: 1px solid #000000;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-size: 10px;
                    font-weight: bold;
                    white-space: nowrap;
                    text-align: center;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
                ">
                    {locality_name}<br>
                    <span style="font-size: 12px;">{formatted_price}</span>
                </div>
            """

            # Create the custom icon
            icon = DivIcon(
                icon_size=(150, 36), # Adjusted size
                icon_anchor=(75, 18), # Center the icon
                html=html
            )
            
            # Add the marker to the map
            folium.Marker(
                location=[row['Lat'], row['Lon']],
                icon=icon
            ).add_to(m)

        # Display the map in Streamlit
        st_folium(m, height=650, width="100%")
        
        # Display summary statistics
        st.subheader("Summary of Filtered Data")
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Median Price", f"${grouped_df['MeanPrice'].mean():,.0f}")
        col2.metric("Min Area Price", f"${min_price:,.0f}")
        col3.metric("Max Area Price", f"${max_price:,.0f}")
        
        st.markdown(f"**Data displayed for {', '.join(selected_types)} in {selected_year}.**")
