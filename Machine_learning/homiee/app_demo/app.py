import os
import random
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import zipfile
import tempfile
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from datetime import datetime

@st.cache_resource
def load_model():
    with zipfile.ZipFile("Machine_learning/homiee/app_demo/model_coordinates.zip", "r") as zip_ref:
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_ref.extractall(tmpdirname)
            model_path = os.path.join(tmpdirname, "model_coordinates.pkl")
            with open(model_path, "rb") as f:
                return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv('Machine_learning/homiee/app_demo/df.csv',
                      usecols=['latitude', 'longitude', 'year', 'quarter', 'Sold_Price'])  # Make sure your data includes year, quarter, lat, lon, actual_price

geolocator = Nominatim(user_agent="myGeocoder")

def get_address(lat, lon):
    try:
        location = geolocator.reverse(f"{lat}, {lon}", timeout=5)
        return location.address if location else "Unknown"
    except:
        return "Unknown"

model = load_model()   
df = load_data()

st.title("House Price Prediction Demo")
mode = st.radio("Choose mode:", ["Show 10 Random Predictions", "Predict on Map Double Click"])

if mode == "Show 10 Random Predictions":
    st.info("Randomly sampled locations from the dataset are shown with actual, predicted prices, and address.")

    sampled = df.sample(10, random_state=random.randint(1, 100))
    m = folium.Map(location=[sampled.latitude.mean(), sampled.longitude.mean()], zoom_start=8)

    # sampled['address'] = get_address(sampled['latitude'], sampled['longitude'])
    sampled['address'] = sampled.apply(lambda row: get_address(row['latitude'], row['longitude']), axis=1)
    sampled['predicted_price'] = model.predict(sampled[model.booster_.feature_name()])
    
    for idx, row in sampled.iterrows():
        popup = (f"Actual price: {row['Sold_Price']}<br>"
                 f"Predicted price: {row['predicted_price']}<br>"
                 f"Address: {row['address']}")
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup, max_width=300),
            tooltip="Click for info"
        ).add_to(m)
    st_folium(m, width=700, height=500)

elif mode == "Predict on Map Double Click":
    # st.info("Double click anywhere on the map to get predicted price for that location.")

    # m = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=11)
    # st.write("Double click anywhere on the map to predict house price for that location (using average year/quarter in dataset).")
    # map_data = st_folium(m, width=700, height=500)

    # if map_data and map_data['last_object_clicked']:
    #     latitude = map_data['last_clicked']['lat']
    #     longitude = map_data['last_clicked']['long']

    #     from datetime import datetime
    #     now = datetime.now()
    #     year = now.year
    #     quarter = (now.month - 1) // 3 + 1

    #     features = np.array([[year, quarter, latitude, longitude]])
    #     pred = model.predict(features)[0]

    #     address = get_address(latitude, longitude)
    #     st.success(f"Predicted price at ({latitude:.4f}, {longitude:.4f}) is **{pred:.2f}**")
    #     st.write(f"Address: {address}")
    st.info("Click anywhere on the map to get the predicted price for that location.")

    # Create a Folium map centered on the mean latitude and longitude of your dataset
    m = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=11)

    # Display the map in Streamlit
    st.write("Click anywhere on the map to predict house price for that location (using current year/quarter).")
    map_data = st_folium(m, width=700, height=500)

    # Check for single-click data from the map interaction
    if map_data and 'last_clicked' in map_data and map_data['last_clicked']:
        latitude = map_data['last_clicked']['lat']
        longitude = map_data['last_clicked']['lng']

        # Get current year and quarter for prediction
        now = datetime.now()
        year = now.year
        quarter = (now.month - 1) // 3 + 1

        # Prepare features for prediction
        features = np.array([[year, quarter, latitude, longitude]])
        pred = model.predict(features)[0]

        # Get address for the clicked location
        address = get_address(latitude, longitude)
        
        # Display prediction and address
        st.success(f"Predicted price at ({latitude:.4f}, {longitude:.4f}) is **${pred:.2f}**")
        st.write(f"Address: {address}")
    else:
        st.warning("Click on the map to select a location for prediction.")
