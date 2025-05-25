import streamlit as st
import pandas as pd
import pickle
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

@st.cache_resource
def load_model():
    with open('your_model.pkl', 'rb') as f:            # Make sure path/filename matches your file
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv('your_data.csv')  # Make sure your data includes year, quarter, lat, lon, actual_price

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

    sampled = df.sample(10, random_state=42)
    m = folium.Map(location=[sampled.lat.mean(), sampled.lon.mean()], zoom_start=11)

    for idx, row in sampled.iterrows():
        features = [[row['year'], row['quarter'], row['lat'], row['lon']]]
        pred = model.predict(features)[0]
        address = get_address(row['lat'], row['lon'])
        popup = (f"Actual price: {row['actual_price']}<br>"
                 f"Predicted price: {pred:.2f}<br>"
                 f"Address: {address}")
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup, max_width=300),
            tooltip="Click for info"
        ).add_to(m)
    st_folium(m, width=700, height=500)

elif mode == "Predict on Map Double Click":
    st.info("Double click anywhere on the map to get predicted price for that location.")

    m = folium.Map(location=[df.lat.mean(), df.lon.mean()], zoom_start=11)
    st.write("Double click anywhere on the map to predict house price for that location (using average year/quarter in dataset).")
    map_data = st_folium(m, width=700, height=500)

    if map_data and map_data['last_object_clicked']:
        lat = map_data['last_clicked']['lat']
        lon = map_data['last_clicked']['lng']
        year = int(df['year'].mode()[0])
        quarter = int(df['quarter'].mode()[0])

        features = [[year, quarter, lat, lon]]
        pred = model.predict(features)[0]
        address = get_address(lat, lon)
        st.success(f"Predicted price at ({lat:.4f}, {lon:.4f}) is **{pred:.2f}**")
        st.write(f"Address: {address}")
