import streamlit as st
from PIL import Image
from io import BytesIO # Keep for completeness, though not strictly needed for the Image.open(uploaded_file) approach

# Import the Hugging Face pipeline
from transformers import pipeline
import torch # Import torch to check for CUDA availability

# --- Model Loading ---
# This function loads the model once and caches it.
@st.cache_resource
def load_classifier_pipeline():
    """
    Loads a standard image classification model from Hugging Face.
    This function will be run only once when the app starts.
    We are switching to a standard classification model as zero-shot requires labels.
    """
    try:
        # Use GPU if available, otherwise CPU
        device = 0 if torch.cuda.is_available() else -1
        st.info(f"Loading standard image classification model on device: {'GPU' if device == 0 else 'CPU'}...")
        
        # Swapping to a standard Image Classification pipeline and model
        model_pipeline = pipeline(
            "image-classification",
            # This is a common, general-purpose image classification model
            model="google/vit-base-patch16-224", 
            device=device
        )
        st.success("Image Classification model loaded successfully!")
        return model_pipeline
    except Exception as e:
        st.error(f"Error loading classifier model: {e}")
        st.warning("Please ensure 'transformers' and a backend (torch/tensorflow) are installed.")
        st.stop()
        return None

classifier = load_classifier_pipeline()

# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Image Classifier")
st.title("🖼️ Standard Image Classifier (No Label List)")

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
        if classifier is None:
            st.error("Model is not loaded. Cannot run classification.")
        else:
            with st.spinner("🧠 Classifying image... Please wait."):
                try:
                    # Run standard image classification on the selected image
                    # The pipeline will return a list of top predictions by default
                    result = classifier(selected_img_to_display)
                    
                    # The result is already a sorted list (by default), 
                    # but we explicitly grab the top prediction
                    top_prediction = result[0]
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
        
        # The key names 'label' and 'score' are consistent with the old zero-shot output
        label = result['label']
        score = result['score']
        
        # Display the predicted label and its confidence score
        st.metric(label="Predicted Class", value=f"`{label}`")
        st.progress(score, text=f"Confidence: {score:.2%}")
    else:
        st.error(result)

if not uploaded_files:
    st.info("Please upload one or more images to begin.")
