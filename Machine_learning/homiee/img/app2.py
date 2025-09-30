import streamlit as st
from PIL import Image
import torch
import json
import io # Used for handling byte data from Streamlit FileUploader

# Import the specific Hugging Face classes for the BLIP VQA model
from transformers import BlipProcessor, BlipForQuestionAnswering

# --- Configuration ---
BLIP_MODEL_NAME = "Salesforce/blip-vqa-base" 

# The structured prompt instructing the VLM to output JSON
VLM_PROMPT_TEMPLATE = """
Based on the image, determine the top 3 most likely room types and their confidence scores (out of 1.0). 
Output the result ONLY as a compact JSON list of objects following this strict format:
[
  { "label": "Living Room", "score": 0.95 },
  { "label": "Bedroom", "score": 0.03 },
  { "label": "Kitchen", "score": 0.02 }
]
"""
# This prompt will be passed directly as the "question" to the BLIP model
FULL_VLM_PROMPT = VLM_PROMPT_TEMPLATE

# --- Model Loading (FIXED for BLIP) ---
@st.cache_resource
def load_blip_vqa_model():
    """
    Loads the explicit BlipProcessor and BlipForQuestionAnswering model.
    """
    try:
        # Determine the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Loading BLIP VQA model: {BLIP_MODEL_NAME} on device: {device}...")

        # Load the dedicated processor and model
        processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
        model = BlipForQuestionAnswering.from_pretrained(BLIP_MODEL_NAME).to(device)
        
        st.success(f"BLIP VQA model loaded successfully on {device}!")
        return processor, model, device

    except Exception as e:
        st.error(f"Error loading BLIP VQA model: {e}")
        st.stop()
        return None, None, None

processor, model, device = load_blip_vqa_model()

# --- BLIP Classification Function ---
def vlm_classifier_structured(raw_image: Image.Image, prompt: str):
    """
    Performs VLM classification using the loaded BLIP model and processes the structured JSON output.
    """
    
    # 1. Prepare inputs using the dedicated processor
    inputs = processor(
        raw_image, 
        prompt, 
        return_tensors="pt"
    ).to(device)

    # 2. Generate the answer text using the model's generate method
    # Use max_new_tokens for the JSON structure
    out = model.generate(
        **inputs, 
        max_new_tokens=200, 
        do_sample=False, 
        temperature=0.0
    )
    
    # 3. Decode the raw output
    generated_text = processor.decode(out[0], skip_special_tokens=True)

    # --- JSON Parsing Logic ---
    json_str = generated_text.strip()
    
    # 1. Robust Prompt Cleanup: Look for the opening bracket '['
    start_index = json_str.find('[')
    if start_index != -1:
        json_str = json_str[start_index:]

    # 2. Clean common VLM JSON wrappers
    if '```json' in json_str:
        json_str = json_str.split('```json')[1].split('```')[0].strip()
    elif '```' in json_str:
        json_str = json_str.split('```')[1].split('```')[0].strip()

    # 3. Handle truncated JSON (often happens if max_new_tokens is too small)
    if json_str.count('{') > json_str.count('}'):
        # Simple fix: if the last element is missing a closing brace, truncate it
        if json_str.endswith(',') or json_str.endswith(']'):
            json_str = json_str.rsplit('}', 1)[0] + '}]'
            
    if not json_str.startswith('['):
        return {"error": f"JSON parsing failed. Raw VLM output was not valid JSON after cleanup. Cleaned Output: {json_str}"}

    # 4. Attempt to parse the cleaned JSON
    try:
        predictions = json.loads(json_str)
        
        if isinstance(predictions, list):
            def get_score(p):
                try:
                    return float(p.get('score', 0.0))
                except ValueError:
                    return 0.0
                    
            predictions = [p for p in predictions if 'label' in p and 'score' in p]
            
            return sorted(predictions, key=get_score, reverse=True)[:3]
        else:
            raise ValueError("VLM response was not a valid JSON list.")

    except Exception as e:
        return {"error": f"JSON parsing failed after initial cleanup. Cleaned Output: {json_str}. Error: {e}"}


# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="BLIP VQA Structured Classifier")
st.title("🧠 BLIP VQA Structured Classifier (Top 3 Scores)")

# --- 1. Load Data: User Image Upload ---
st.header("1. Upload Your Images")
uploaded_files = st.file_uploader(
    "Choose one or more image files",
    type=['png', 'jpg', 'jpeg', 'webp'],
    accept_multiple_files=True
)

# Initialize session state variables
if 'selected_image_data' not in st.session_state:
    st.session_state.selected_image_data = None
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None
if 'last_uploaded_files' not in st.session_state:
    st.session_state.last_uploaded_files = []

# Clear selection logic
if uploaded_files and (st.session_state.last_uploaded_files != uploaded_files):
    st.session_state.selected_image_data = None
    st.session_state.classification_result = None
    st.session_state.last_uploaded_files = uploaded_files

# --- 2. Display Images ---
if uploaded_files:
    st.header("2. Select an Image for Classification")
    st.write("Click a button below an image to select it.")
    cols_per_row = 5
    image_cols = st.columns(cols_per_row)
    for i, uploaded_file in enumerate(uploaded_files):
        with image_cols[i % cols_per_row]:
            try:
                # Need to read the file into memory buffer before opening with PIL
                img_data = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')
                uploaded_file.seek(0) # Reset pointer for potential re-read
                st.image(img_data, caption=f"{uploaded_file.name[:20]}...", width=150)
                if st.button(f"Select Image {i+1}", key=f"select_btn_{i}"):
                    # Store the uploaded_file object which is memory-efficient
                    st.session_state.selected_image_data = uploaded_file
                    st.session_state.classification_result = None
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

# --- 3. Classify Selected Image ---
if st.session_state.selected_image_data:
    st.markdown("---")
    st.header("3. Classify Selected Image using Structured Prompt")
    st.write(f"Model: **{BLIP_MODEL_NAME}** running on **{device}**")
    st.write("The BLIP VQA model is prompted to output a **JSON list** of the top 3 room types.")
    
    selected_file = st.session_state.selected_image_data
    st.write(f"**Selected Image:** `{selected_file.name}`")
    
    # Read the image again for display and processing
    selected_file.seek(0)
    selected_img_to_display = Image.open(io.BytesIO(selected_file.read())).convert('RGB')
    st.image(selected_img_to_display, caption="Image to be classified", width=400)

    if st.button("🚀 Run BLIP VQA Classification (Top 3)"):
        if model is None:
            st.error("Model is not loaded. Cannot run classification.")
        else:
            with st.spinner("🧠 Generating structured predictions... Please wait."):
                try:
                    # Pass the PIL Image directly to the VLM function
                    result = vlm_classifier_structured(selected_img_to_display, prompt=FULL_VLM_PROMPT)
                    st.session_state.classification_result = result
                except Exception as e:
                    st.error(f"An unexpected error occurred during VLM classification: {e}")
                    st.session_state.classification_result = {"error": f"Classification failed ({e})"}
            st.rerun()

# --- 4. Show Result ---
if st.session_state.classification_result:
    result = st.session_state.classification_result
    
    if isinstance(result, dict) and 'error' in result:
        st.error("Classification Error")
        st.code(result['error'])
    elif isinstance(result, list):
        st.success("✅ BLIP Generation and Parsing Complete!")
        st.subheader("Top 3 Predicted Room Types")
        for i, prediction in enumerate(result):
            label = prediction.get('label', 'N/A')
            score = prediction.get('score', 0.0)
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
            
            st.markdown(f"**#{i+1}: {label}**")
            st.progress(score, text=f"Confidence: {score:.2%}")
            
        st.markdown("**Important Note:** These 'scores' are generated by the VLM's text output and are **not** true logit probabilities. The model is guessing a number based on the prompt's instructions.")
    else:
        st.error("Received an unexpected result format from the VLM.")

if not uploaded_files:
    st.info("Please upload one or more images to begin.")
