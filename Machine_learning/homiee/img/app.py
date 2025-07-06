import streamlit as st
import pandas as pd
import numpy as np # Still useful for potential NaN handling if any creeps in elsewhere
import re # Still useful for column pattern in data processing function, though not used now
import json
from PIL import Image
import requests
from io import BytesIO
import ast # Import for literal_eval

# Import the Hugging Face pipeline
from transformers import pipeline
import torch # Import torch to check for CUDA availability

# --- Data Loading (Streamlined) ---

# Cache the DataFrame loading to avoid re-reading the CSV on every rerun
@st.cache_data
def load_and_preprocess_dataframe(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # Ensure 'image_urls' column exists and convert its string representation to actual lists
        if 'image_urls' in df.columns:
            # Use ast.literal_eval to safely convert string representations of lists to actual lists
            # Apply to non-null values to avoid errors on NaNs
            df['image_urls'] = df['image_urls'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
            # Filter out rows where image_urls list is empty after conversion
            df_filtered = df[df['image_urls'].apply(len) > 0].copy()
            st.success(f"DataFrame loaded and 'image_urls' column converted. {len(df_filtered)} rows with images found.")
            return df_filtered
        else:
            st.error(f"Error: 'image_urls' column not found in {file_path}. Please ensure your CSV is preprocessed.")
            st.stop()
            return pd.DataFrame()

    except FileNotFoundError:
        st.error(f"Error: CSV file not found at {file_path}")
        st.stop() # Stop the app if data is missing
    except Exception as e:
        st.error(f"Error loading or processing DataFrame from {file_path}: {e}")
        st.stop()
    return pd.DataFrame() # Return empty if error

# Load the preprocessed DataFrame directly
df_processed_and_filtered = load_and_preprocess_dataframe('Machine_learning/homiee/img/df_img_st_demo.csv')

# --- Model Loading ---
# Initialize the classifier pipeline using st.cache_resource
@st.cache_resource
def load_classifier_pipeline():
    """
    Loads the zero-shot image classification model.
    This function will be run only once when the app starts.
    """
    try:
        # Use GPU if available, otherwise CPU
        device = 0 if torch.cuda.is_available() else -1
        st.info(f"Loading classifier model on device: {'GPU' if device == 0 else 'CPU'}...")
        model_pipeline = pipeline(
            "zero-shot-image-classification",
            model="strollingorange/roomLuxuryAnnotater",
            device=device
        )
        st.success("Classifier model loaded successfully!")
        return model_pipeline
    except Exception as e:
        st.error(f"Error loading classifier model: {e}")
        st.warning("Please ensure 'transformers' and a backend (torch/tensorflow) are installed.")
        st.stop() # Stop the app if model loading fails
        return None

classifier = load_classifier_pipeline()

# --- Candidate Labels Loading ---
# Cache the candidate labels loading from JSON
@st.cache_data
def load_candidate_labels(file_path):
    try:
        with open(file_path, 'r') as json_file:
            labels = json.load(json_file)
        if not isinstance(labels, list):
            st.error(f"Error: JSON file at {file_path} does not contain a list.")
            st.stop()
        return labels
    except FileNotFoundError:
        st.error(f"Error: Candidate labels JSON file not found at {file_path}")
        st.stop()
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {file_path}. Check file format.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading candidate labels from {file_path}: {e}")
        st.stop()
    return []

composite_candidate_labels = load_candidate_labels('Machine_learning/homiee/img/composite_candidate_labels.json')

# --- Utility Function for Image Display ---
@st.cache_data(show_spinner=False) # Cache downloaded images
def get_image_from_url(url):
    """Downloads an image from a URL and returns a PIL Image object."""
    try:
        response = requests.get(url, timeout=10) # Increased timeout for robustness
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        img = Image.open(BytesIO(response.content))
        return img
    except requests.exceptions.RequestException as e:
        st.error(f"Could not load image from {url}: Network or URL error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred processing image from {url}: {e}")
        return None


# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Image Classifier Demo")
st.title("Image Selection and Zero-Shot Classification")

# 1. Select a row from df_processed_and_filtered
st.header("1. Select an Item ID")

# Check if df_processed_and_filtered is empty before proceeding
if df_processed_and_filtered.empty:
    st.warning("No data found or processed. Please check your CSV and file paths.")
else:
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

        # Using st.session_state to manage selected image and classification result
        if 'selected_image_url' not in st.session_state:
            st.session_state.selected_image_url = None
        if 'classification_result' not in st.session_state:
            st.session_state.classification_result = None
        if 'last_selected_id' not in st.session_state:
            st.session_state.last_selected_id = None

        # Clear result if a new ID is selected
        if st.session_state.last_selected_id != selected_id:
            st.session_state.classification_result = None
            st.session_state.selected_image_url = None
            st.session_state.last_selected_id = selected_id # Update last selected ID

        st.write("Click on an image to select it for classification:")

        # Display images in a grid
        cols_per_row = 5 # Adjust as needed for desired miniature size

        # Create columns dynamically
        image_cols = st.columns(cols_per_row)

        image_counter = 0
        for i, url in enumerate(image_urls_for_selected_id):
            with image_cols[image_counter % cols_per_row]:
                # Download and display the image
                img_data = get_image_from_url(url)
                if img_data:
                    st.image(img_data, caption=f"Image {i+1}", width=120) # Miniature width

                    # Create a button for selection
                    if st.button(f"Select Image {i+1}", key=f"select_btn_{selected_id}_{i}"):
                        st.session_state.selected_image_url = url
                        st.session_state.classification_result = None # Clear previous result
                        st.rerun() # Rerun to display selection and clear old result
                else:
                    st.write(f"Image {i+1} (Error loading)")
            image_counter += 1

        # Display selected image and classification button
        if st.session_state.selected_image_url:
            st.markdown("---")
            st.subheader("3. Classify Selected Image")
            st.write(f"**Selected Image URL:** `{st.session_state.selected_image_url}`")

            selected_img_to_display = get_image_from_url(st.session_state.selected_image_url)
            if selected_img_to_display:
                st.image(selected_img_to_display, caption="Currently Selected Image", width=300)

            # Button to trigger classification
            if st.button("Run Classifier"):
                if classifier is None: # Check if classifier was loaded successfully
                    st.error("Classifier model is not loaded. Cannot run classification.")
                elif not composite_candidate_labels: # Check if candidate labels were loaded
                    st.error("Candidate labels not loaded. Cannot run classification.")
                else:
                    with st.spinner("Classifying image..."):
                        try:
                            # Download the image content using requests
                            response = requests.get(st.session_state.selected_image_url, timeout=15)
                            response.raise_for_status()

                            # Open the image using Pillow from the binary content
                            image_for_classification = Image.open(BytesIO(response.content))

                            # Run zero-shot classification using the loaded pipeline
                            result = classifier(image_for_classification, candidate_labels=composite_candidate_labels)

                            # Output the top label
                            top_label = sorted(result, key=lambda x: x['score'], reverse=True)[0]['label']
                            st.session_state.classification_result = top_label

                        except requests.exceptions.RequestException as e:
                            st.error(f"Error downloading image for classification: {e}")
                            st.session_state.classification_result = f"Error: Image download failed ({e})"
                        except Exception as e:
                            st.error(f"An unexpected error occurred during classification: {e}")
                            st.session_state.classification_result = f"Error: Classification failed ({e})"
                    st.success("Classification Complete!")
                    st.rerun() # Rerun to display the result

            # Display classification result
            if st.session_state.classification_result:
                st.write("---")
                st.success("Classification Result:")
                st.write(f"**Predicted Label:** `{st.session_state.classification_result}`")
        else:
            st.markdown("---")
            st.info("Select an image above to run the classifier.")
