import streamlit as st
import json
from PIL import Image
from io import BytesIO

# Import the Hugging Face pipeline
from transformers import pipeline
import torch # Import torch to check for CUDA availability

# --- Model Loading ---
# This function loads the model once and caches it.
@st.cache_resource
def load_classifier_pipeline():
    """
    Loads the zero-shot image classification model from Hugging Face.
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
        st.stop()
        return None

classifier = load_classifier_pipeline()

# --- Candidate Labels Loading ---
# This function loads the possible labels from a JSON file and caches them.
@st.cache_data
def load_candidate_labels(file_path):
    """Loads a list of candidate labels from a JSON file."""
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
        st.error(f"An error occurred loading labels from {file_path}: {e}")
        st.stop()
    return []

# NOTE: You'll need to create this JSON file or update the path.
# Example 'composite_candidate_labels.json': ["modern", "classic", "rustic", "minimalist"]
composite_candidate_labels = load_candidate_labels('Machine_learning/homiee/img/composite_candidate_labels.json')

# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Zero-Shot Image Classifier")
st.title("📸 Zero-Shot Image Classifier")

# --- 1. Load Data: User Image Upload ---
st.header("1. Upload Your Images")
uploaded_files = st.file_uploader(
    "Choose one or more image files",
    type=['png', 'jpg', 'jpeg', 'webp'],
    accept_multiple_files=True
)

# Initialize session state variables to manage the app's flow
if 'selected_image_data' not in st.session_state:
    st.session_state.selected_image_data = None
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None
if 'last_uploaded_files' not in st.session_state:
    st.session_state.last_uploaded_files = []

# Clear selection if the user uploads a new set of files
if uploaded_files and (st.session_state.last_uploaded_files != uploaded_files):
    st.session_state.selected_image_data = None
    st.session_state.classification_result = None
    st.session_state.last_uploaded_files = uploaded_files

# --- 2. Display Images ---
if uploaded_files:
    st.header("2. Select an Image for Classification")
    st.write("Click a button below an image to select it.")

    # Display images in a grid
    cols_per_row = 5
    image_cols = st.columns(cols_per_row)

    for i, uploaded_file in enumerate(uploaded_files):
        with image_cols[i % cols_per_row]:
            try:
                img_data = Image.open(uploaded_file)
                st.image(img_data, caption=f"{uploaded_file.name[:20]}...", width=150)

                # Create a "Select" button for each image
                if st.button(f"Select Image {i+1}", key=f"select_btn_{i}"):
                    st.session_state.selected_image_data = uploaded_file
                    st.session_state.classification_result = None # Clear previous result
                    st.rerun()

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

# --- 3. Classify Selected Image ---
if st.session_state.selected_image_data:
    st.markdown("---")
    st.header("3. Classify Selected Image")

    selected_file = st.session_state.selected_image_data
    st.write(f"**Selected Image:** `{selected_file.name}`")

    # Display the selected image in a larger format
    selected_img_to_display = Image.open(selected_file)
    st.image(selected_img_to_display, caption="Image to be classified", width=400)

    # Button to trigger the classification process
    if st.button("🚀 Run Classifier"):
        if classifier is None or not composite_candidate_labels:
            st.error("Model or labels are not loaded. Cannot run classification.")
        else:
            with st.spinner("🧠 Classifying image... Please wait."):
                try:
                    # Run zero-shot classification on the selected image
                    result = classifier(selected_img_to_display, candidate_labels=composite_candidate_labels)
                    # Store the prediction with the highest score
                    top_prediction = sorted(result, key=lambda x: x['score'], reverse=True)[0]
                    st.session_state.classification_result = top_prediction
                except Exception as e:
                    st.error(f"An unexpected error occurred during classification: {e}")
                    st.session_state.classification_result = f"Error: Classification failed ({e})"
            st.rerun()

# --- 4. Show Result ---
if st.session_state.classification_result:
    result = st.session_state.classification_result
    
    # Check if the result is a dictionary (successful prediction) or a string (error)
    if isinstance(result, dict):
        st.success("✅ Classification Complete!")
        st.subheader("Classification Result")
        
        label = result['label']
        score = result['score']
        
        # Display the predicted label and its confidence score
        st.metric(label="Predicted Label", value=f"`{label}`")
        st.progress(score, text=f"Confidence: {score:.2%}")
    else:
        st.error(result)

if not uploaded_files:
    st.info("Please upload one or more images to begin.")
