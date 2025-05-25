import os
import streamlit as st
import pandas as pd
import pickle
import zipfile
import tempfile
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

MODEL_ZIP = "Machine_learning/homiee/app_demo/model_coordinates.zip"
# MODEL_PKL = "Machine_learning/homiee/app_demo/model_coordinates.pkl"

# def extract_model(zip_path=MODEL_ZIP, model_path=MODEL_PKL):
#     if not os.path.isfile(model_path):
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall()
#         print(f"Extracted {model_path} from {zip_path}")

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
    return pd.read_csv('Machine_learning/homiee/app_demo/df.csv')  # Make sure your data includes year, quarter, lat, lon, actual_price

geolocator = Nominatim(user_agent="myGeocoder")

def get_address(lat, lon):
    try:
        location = geolocator.reverse(f"{lat}, {lon}", timeout=5)
        return location.address if location else "Unknown"
    except:
        return "Unknown"

model = load_model()
# st.write("Model object:", model)
# if model is None:
#     st.error("Model failed to load!")   
df = load_data()

st.title("House Price Prediction Demo")

mode = st.radio("Choose mode:", ["Show 10 Random Predictions", "Predict on Map Double Click"])

if mode == "Show 10 Random Predictions":
    st.info("Randomly sampled locations from the dataset are shown with actual, predicted prices, and address.")

    sampled = df.sample(10, random_state=42)
    m = folium.Map(location=[sampled.latitude.mean(), sampled.longitude.mean()], zoom_start=11)

    for idx, row in sampled.iterrows():
        features = [[row['year'], row['quarter'], row['latitude'], row['longitude']]]
        pred = model.predict(features)[0]
        address = get_address(row['latitude'], row['longitude'])
        popup = (f"Actual price: {row['actual_price']}<br>"
                 f"Predicted price: {pred:.2f}<br>"
                 f"Address: {address}")
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup, max_width=300),
            tooltip="Click for info"
        ).add_to(m)
    st_folium(m, width=700, height=500)

elif mode == "Predict on Map Double Click":
    st.info("Double click anywhere on the map to get predicted price for that location.")

    m = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=11)
    st.write("Double click anywhere on the map to predict house price for that location (using average year/quarter in dataset).")
    map_data = st_folium(m, width=700, height=500)

    if map_data and map_data['last_object_clicked']:
        latitude = map_data['last_clicked']['lat']
        longitude = map_data['last_clicked']['long']

        from datetime import datetime
        now = datetime.now()
        year = now.year
        quarter = (now.month - 1) // 3 + 1

        features = [[year, quarter, latitude, longitude]]
        pred = model.predict(features)[0]
        address = get_address(latitude, longitude)
        st.success(f"Predicted price at ({latitude:.4f}, {longitude:.4f}) is **{pred:.2f}**")
        st.write(f"Address: {address}")
