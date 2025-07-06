import streamlit as st
import pandas as pd
import numpy as np
import re
from PIL import Image # For image display if using local files or PIL operations
import requests # For downloading images to display them
from io import BytesIO # To handle image bytes from URLs

# --- 0. Mock Data and Model (Replace with your actual data and model) ---

# Re-create the data processing function to ensure it's available in the Streamlit app
def consolidate_image_urls_and_filter_empty_rows(df_input):
    image_col_pattern = r'propertyDetails\.media\.images\[\d+\]\.src'
    image_columns = [col for col in df_input.columns if re.match(image_col_pattern, col)]

    def get_non_nan_urls(row):
        urls = []
        for col_name in image_columns:
            url = row.get(col_name)
            if pd.notna(url) and isinstance(url, str) and url.strip():
                urls.append(url.strip())
        return urls

    df_input['image_urls'] = df_input.apply(get_non_nan_urls, axis=1)
    df_filtered = df_input[df_input['image_urls'].apply(len) > 0].copy()
    return df_filtered[['Id', 'image_urls']]

df_original = pd.DataFrame(data)

# Process the DataFrame to get df_processed_and_filtered
df_processed_and_filtered = consolidate_image_urls_and_filter_empty_rows(df_original.copy())

# Mock Classifier Model (Replace with your actual model loading and prediction)
class MockClassifier:
    def __init__(self, name="Mock Model"):
        self.name = name

    def predict(self, image_url):
        st.info(f"Classifier '{self.name}' is analyzing image from: {image_url}")
        # Simulate some processing time
        import time
        time.sleep(1)
        # Return a dummy result based on the URL or some random logic
        if "FF0000" in image_url:
            return "Red Dominant"
        elif "00FF00" in image_url:
            return "Green Dominant"
        elif "0000FF" in image_url:
            return "Blue Dominant"
        else:
            return "Mixed Colors"

# Initialize your classifier (load it once)
@st.cache_resource # Use st.cache_resource to load models only once
def load_classifier():
    return MockClassifier()

classifier = load_classifier()

# --- Utility Function for Image Display ---
@st.cache_data(show_spinner=False) # Cache downloaded images
def get_image_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except requests.exceptions.RequestException as e:
        st.error(f"Could not load image from {url}: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred processing image from {url}: {e}")
        return None


# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Image Classifier Demo")
st.title("Image Selection and Classification Demo")

# 1. Select a row from df_processed_and_filtered
st.header("1. Select an Item ID")

# Create a list of 'Id's for the selectbox
item_ids = df_processed_and_filtered['Id'].tolist()
selected_id = st.selectbox(
    "Choose an Item ID to view its images:",
    options=item_ids,
    index=0 # Default to the first ID
)

# Get the selected row
selected_row = df_processed_and_filtered[df_processed_and_filtered['Id'] == selected_id].iloc[0]
image_urls_for_selected_id = selected_row['image_urls']

if not image_urls_for_selected_id:
    st.warning(f"No images found for Item ID {selected_id}.")
else:
    st.subheader(f"Images for Item ID: {selected_id}")

    # 2. Display images from URLs in tile/miniature style with selection
    # We'll use columns to create a grid layout
    num_images = len(image_urls_for_selected_id)
    # Determine number of columns dynamically, max 5 or 6 for a compact view
    cols_per_row = min(num_images, 5) if num_images > 0 else 1
    
    # Using st.session_state to manage selected image
    if 'selected_image_url' not in st.session_state:
        st.session_state.selected_image_url = None

    if 'classification_result' not in st.session_state:
        st.session_state.classification_result = None

    # Clear result if a new ID is selected
    if st.session_state.get('last_selected_id') != selected_id:
        st.session_state.classification_result = None
        st.session_state.selected_image_url = None
        st.session_state.last_selected_id = selected_id # Update last selected ID

    st.write("Click on an image to select it for classification:")

    # Create columns for image display
    # Use a fixed width for miniatures or let Streamlit auto-size
    miniature_width = 120 # pixels

    image_counter = 0
    # Create columns dynamically to avoid making too many if few images
    image_cols = st.columns(cols_per_row)

    for i, url in enumerate(image_urls_for_selected_id):
        # Place image in the current column
        with image_cols[image_counter % cols_per_row]:
            # Use st.container to group elements if needed, or just directly use st.image and st.button
            
            # Download and display the image
            img_data = get_image_from_url(url)
            if img_data:
                st.image(img_data, caption=f"Image {i+1}", width=miniature_width)
                
                # Create a button for selection. Key is important for uniqueness
                if st.button(f"Select Image {i+1}", key=f"select_btn_{selected_id}_{i}"):
                    st.session_state.selected_image_url = url
                    st.session_state.classification_result = None # Clear previous result
                    st.experimental_rerun() # Rerun to display selection and clear old result
            else:
                st.write(f"Image {i+1} (Error)")
        image_counter += 1

    # Display selected image and classification button
    if st.session_state.selected_image_url:
        st.markdown("---")
        st.subheader("3. Classify Selected Image")
        st.write(f"**Selected Image URL:** {st.session_state.selected_image_url}")

        selected_img_to_display = get_image_from_url(st.session_state.selected_image_url)
        if selected_img_to_display:
            st.image(selected_img_to_display, caption="Currently Selected Image", width=300)

        # Button to trigger classification
        if st.button("Run Classifier"):
            with st.spinner("Classifying image..."):
                st.session_state.classification_result = classifier.predict(st.session_state.selected_image_url)
            st.success("Classification Complete!")
            st.experimental_rerun() # Rerun to display the result

        # Display classification result
        if st.session_state.classification_result:
            st.write("---")
            st.success("Classification Result:")
            st.write(f"**Result:** `{st.session_state.classification_result}`")
    else:
        st.markdown("---")
        st.info("Select an image above to run the classifier.")
