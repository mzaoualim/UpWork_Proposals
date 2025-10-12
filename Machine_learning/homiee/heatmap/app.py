import streamlit as st
import pandas as pd
import folium
import json
import io
import branca.colormap as cm 
from streamlit_folium import st_folium 
from folium.features import GeoJsonTooltip, GeoJson, DivIcon 
import numpy as np

# --- 1. CONFIGURATION AND DATA LOADING ---

st.set_page_config(
    page_title="Australian Property Median Price Map",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Coordinates for centering the map on the general area covered by the GeoJSON
# Based on the coordinates in boundaries.geojson (Sydney/NSW area)
MAP_CENTER = (-33.95, 151.00) 
MAP_ZOOM = 14

@st.cache_data
def load_data():
    """
    Loads the GeoJSON boundaries and CSV property data using standard file I/O.
    """
    try:
        # Load GeoJSON boundaries using standard Python file I/O
        with open("Machine_learning/homiee/heatmap/boundaries.geojson", "r") as f:
            boundaries = json.load(f)
        
        # Load CSV data directly using pandas read_csv from file path
        data = pd.read_csv("Machine_learning/homiee/heatmap/median_5loc_df.csv")
        
        # Data Cleaning and Column Preparation
        data.columns = [col.strip() for col in data.columns]
        data.rename(columns={
            'Locality Name': 'Locality',
            'Median Price': 'MedianPrice',
            'Percent Change': 'PercentChange',
            'Type': 'PropertyType'
        }, inplace=True)

        # Convert PercentChange to percentage for better display (e.g., 0.02 to 2.0%)
        # Note: We keep the float for calculations, but the display will be %
        data['PercentChange_Display'] = data['PercentChange'] * 100
        data['MedianPrice'] = data['MedianPrice'].astype(int)
        
        return boundaries, data

    except Exception as e:
        st.error(f"Error loading or processing data files: {e}")
        st.info("Please ensure 'boundaries.geojson' and 'median_5loc_df.csv' are uploaded and readable in the environment.")
        return None, None

boundaries, df = load_data()

# --- 2. UTILITY FUNCTIONS FOR COLORMAPS AND GEOMETRY ---

# Define explicit color palettes to avoid 'AttributeError' with named palettes
# Sequential Palette (Yellow-Green-Blue) for Median Price
SEQ_COLORS = ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494'] 
# Diverging Palette (Red-White-Green) for Percent Change
DIV_COLORS = ['#d73027', '#fc8d59', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4'] 


def create_color_map(metric, data_series):
    """
    Creates a Folium-compatible colormap based on the selected metric.
    """
    if data_series.empty or data_series.isnull().all():
        # Return a simple map if data is empty or all NaN
        return cm.LinearColormap(colors=['#ccc', '#ccc'], vmin=0, vmax=1, caption='No Data')

    # 1. Median Price (Sequential - Blue)
    if metric == 'Median Price':
        min_val, max_val = data_series.min(), data_series.max()
        if min_val == max_val: 
             # Adjust scale slightly if all values are identical
             min_val, max_val = min_val * 0.9, max_val * 1.1 if min_val != 0 else -1, 1

        colormap = cm.LinearColormap(
            colors=SEQ_COLORS,
            vmin=min_val,
            vmax=max_val,
            caption='Median Property Price (AUD)'
        )
        return colormap
    
    # 2. Percent Change (Diverging - Red/Yellow/Green)
    elif metric == 'Percent Change':
        # Find the max absolute deviation from zero for a balanced diverging scale
        max_abs = max(abs(data_series.min()), abs(data_series.max()))
        
        # Ensure vmin and vmax are symmetrical around zero
        min_scale = -max_abs
        max_scale = max_abs

        colormap = cm.LinearColormap(
            colors=DIV_COLORS,
            vmin=min_scale,
            vmax=max_scale,
            caption='Year-over-Year Percent Change (%)'
        )
        return colormap

def get_polygon_center(geometry):
    """
    Calculates the approximate center (bounding box center) of a Folium GeoJSON polygon.
    Returns the center in [lat, lon] format expected by Folium.
    """
    if geometry['type'] == 'Polygon':
        coords = geometry['coordinates'][0] # Assuming simple polygon (no holes)
    elif geometry['type'] == 'MultiPolygon':
        # Simplify: just use the first polygon's coordinates
        coords = geometry['coordinates'][0][0] 
    else:
        return MAP_CENTER # Return map center as fallback

    # Calculate bounding box
    all_lons = [p[0] for p in coords]
    all_lats = [p[1] for p in coords]

    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)

    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    
    # Folium uses [latitude, longitude] order
    return [center_lat, center_lon]


# --- 3. STREAMLIT APP LAYOUT AND FILTERS ---

st.title("🗺️ Australian Property Median Price Map")

# Sidebar for user filtering
with st.sidebar:
    st.header("Data & Map Filters")

    if df is not None:
        # Filter 1: Metric Selection (Key change)
        metric_options = ['Median Price', 'Percent Change']
        selected_metric = st.radio(
            "Select Metric to Visualize",
            options=metric_options,
            index=0, # Default to Median Price
        )
        
        # Filter 2: Year
        all_years = sorted(df['Year'].unique())
        selected_year = st.slider(
            "Select Year",
            min_value=min(all_years),
            max_value=max(all_years),
            value=max(all_years),
            step=1
        )

        # Filter 3: Property Type
        all_types = sorted(df['PropertyType'].unique())
        # Default to the first one available, or prompt user if list is empty
        default_type = all_types[0] if all_types else None
        selected_type = st.selectbox(
            "Select Property Type",
            options=all_types,
            index=all_types.index(default_type) if default_type in all_types else 0,
        )
        
        # Filter 4: Locality Names (Default to all)
        all_localities = sorted(df['Locality'].unique())
        
        # CRITICAL: Default to ALL localities as requested
        selected_localities = st.multiselect(
            "Select Localities",
            options=['PANANIA', 'REVESBY', 'PADSTOW'],
            default=['PANANIA', 'REVESBY', 'PADSTOW'],
            placeholder="Choose localities to display"
        )
        
    st.markdown("""---""")
    st.markdown("**Visualization Note**")
    st.info("The map colors the local government area boundaries based on the selected property metric.")

# --- 4. FILTERING DATA AND PREPARING MAP DATA ---

if df is not None and boundaries is not None:
    if not selected_localities:
        st.error("Please select at least one Locality to display data.")
    else:
        # Apply all filters
        filtered_df = df[
            (df['Year'] == selected_year) &
            (df['Locality'].isin(selected_localities)) &
            (df['PropertyType'] == selected_type)
        ].copy()

        # Select the correct data column based on the selected metric
        if selected_metric == 'Median Price':
            data_column = 'MedianPrice'
            tooltip_column = 'MedianPrice'
        else: # Percent Change
            data_column = 'PercentChange' # Use raw float for Colormap scaling
            tooltip_column = 'PercentChange_Display' # Use the % display column for tooltips


        # Create the mapping dictionary for coloring: {Locality Name: Raw Value}
        mapping_data = filtered_df.set_index('Locality')[data_column].to_dict()

        # Create a formatted data map for the Tooltip: {Locality Name: Formatted Value String}
        def format_value(row):
            val = row[tooltip_column]
            if selected_metric == 'Median Price':
                # Use int for formatting as it was converted earlier
                return f"${int(val):,.0f}" 
            else:
                # Format positive changes with a '+' sign
                sign = '+' if val > 0 and selected_metric == 'Percent Change' else ''
                return f"{sign}{val:.2f}%"

        tooltip_data_map = filtered_df.set_index('Locality').apply(
            format_value, axis=1
        ).to_dict()


        # Filter GeoJSON features to only include selected localities
        filtered_features = [
            feature for feature in boundaries['features']
            if feature['properties'].get('name') in selected_localities
        ]
        
        # Create a mutable GeoJSON object
        filtered_geojson = {
            'type': 'FeatureCollection',
            'features': filtered_features
        }
        
        # --- CRITICAL: Merge data into GeoJSON properties for Coloring and Tooltip ---
        for feature in filtered_geojson['features']:
            locality = feature['properties'].get('name')
            
            # Add the raw value (for coloring) and formatted value (for tooltip/label)
            if locality in mapping_data:
                feature['properties']['raw_value'] = mapping_data[locality]
                feature['properties']['formatted_value'] = tooltip_data_map[locality]
            else:
                # Set defaults for non-matched or missing data points
                feature['properties']['raw_value'] = None
                feature['properties']['formatted_value'] = 'N/A'


        # --- 5. MAP GENERATION LOGIC (GeoJson with Labels) ---

        m = folium.Map(
            location=MAP_CENTER, 
            zoom_start=MAP_ZOOM,
            tiles="CartoDB positron" 
        )

        # Create the color map and add it to the map
        colormap = create_color_map(selected_metric, filtered_df[data_column])
        m.add_child(colormap)

        # Function to style the GeoJson (color based on raw_value property)
        def style_function(feature):
            value = feature['properties'].get('raw_value')
            fill_color = colormap(value) if value is not None else '#ccc'
            
            return {
                'fillColor': fill_color,
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7
            }
        
        # Add the main GeoJson layer (coloring + hover tooltip)
        GeoJson(
            filtered_geojson,
            name="Property Data",
            style_function=style_function,
            tooltip=GeoJsonTooltip(
                fields=['name', 'formatted_value'], 
                aliases=['Locality:', f'{selected_metric}:'],
                localize=True,
                sticky=True,
                labels=True,
                style="""
                    background-color: #F0F0F0;
                    color: #444444;
                    font-family: sans-serif;
                    font-size: 14px;
                    padding: 6px;
                    border: 1px solid #aaa;
                    box-shadow: 2px 2px 3px rgba(0,0,0,0.2);
                """,
            ),
            highlight_function=lambda x: {'weight': 3, 'color': 'white', 'fillOpacity': 0.9}
        ).add_to(m)

        
        # --- NEW: Add static text labels (DivIcon Markers) ---
        for feature in filtered_geojson['features']:
            locality = feature['properties'].get('name')
            formatted_value = feature['properties'].get('formatted_value')
            center = get_polygon_center(feature['geometry'])

            if formatted_value != 'N/A':
                # Determine text color based on metric for better contrast
                # text_color = 'green' if (selected_metric == 'Percent Change' and feature['properties']['raw_value'] > 0) else 'red' if (selected_metric == 'Percent Change' and feature['properties']['raw_value'] < 0) else '#333'
                text_color = 'blue'
                
                # HTML content for the label
                html_label = f"""
                <div style="text-align: center; white-space: nowrap; font-weight: bold; font-size: 10px; color: {text_color}; text-shadow: 0 0 2px white, 0 0 2px white; mix-blend-mode: difference;">
                    {locality}<br>
                    {formatted_value}
                </div>
                """
                
                # Create the DivIcon marker at the center of the locality
                folium.Marker(
                    location=center,
                    icon=DivIcon(
                        icon_size=(150, 36), # Size of the HTML container
                        icon_anchor=(75, 18), # Center the icon at the location
                        html=html_label
                    )
                ).add_to(m)


        # Display the map in Streamlit
        st_folium(m, height=650, width="100%")
        
        # Display summary statistics
        st.subheader("Summary of Filtered Data")
        
        if not filtered_df.empty:
            mean_val = filtered_df[data_column].mean()
            min_val = filtered_df[data_column].min()
            max_val = filtered_df[data_column].max()
            
            # Determine how to format the summary metrics
            if selected_metric == 'Median Price':
                mean_str = f"${mean_val:,.0f}"
                min_str = f"${min_val:,.0f}"
                max_str = f"${max_val:,.0f}"
            else:
                # Use the display column for percentage
                mean_display = filtered_df['PercentChange_Display'].mean()
                min_display = filtered_df['PercentChange_Display'].min()
                max_display = filtered_df['PercentChange_Display'].max()

                mean_str = f"{mean_display:.2f}%"
                min_str = f"{min_display:.2f}%"
                max_str = f"{max_display:.2f}%"

            col1, col2, col3 = st.columns(3)
            col1.metric(f"Average {selected_metric}", mean_str)
            col2.metric(f"Min Area {selected_metric}", min_str)
            col3.metric(f"Max Area {selected_metric}", max_str)
            
            st.markdown(f"**Data displayed for {selected_type} in {selected_year}.**")
        else:
            st.warning(f"No data points found for {selected_type} in {selected_year} in the selected localities.")
