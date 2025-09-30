import streamlit as st
from PIL import Image
import torch
import json
from transformers import pipeline

# --- Configuration (Unchanged) ---
VLM_MODEL_NAME = "Salesforce/blip-vqa-base" 

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
FULL_VLM_PROMPT = f"{VLM_PROMPT_TEMPLATE}"

# --- Model Loading (Unchanged) ---
@st.cache_resource
def load_vlm_pipeline():
    """
    Loads a Vision-Language Model (VLM) for structured output generation 
    using the dedicated 'image-to-text' pipeline.
    """
    try:
        device = 0 if torch.cuda.is_available() else -1
        st.info(f"Loading VLM model: {VLM_MODEL_NAME} on device: {'GPU' if device == 0 else 'CPU'}...")

        model_pipeline = pipeline(
            "image-to-text",
            model=VLM_MODEL_NAME,
            device=device 
        )
        
        # --- Wrapper Function (FIXED for JSON parsing) ---
        def vlm_classifier_structured(image, prompt):
            """Runs VLM generation and attempts to clean the structured JSON output."""
            
            # Run the pipeline with max tokens for JSON
            result = model_pipeline(
                image, 
                prompt=prompt, 
                max_new_tokens=200, 
            )
            generated_text = result[0]['generated_text']

            # --- 🎯 FIX: Robust Prompt Cleanup ---
            # 1. Clean up generated text (strip newlines, spaces, etc.)
            json_str = generated_text.strip()
            
            # 2. Heuristically remove the echoed prompt, if present.
            # The prompt starts with "Based on the image..."
            # We look for the closing structure of the prompt: "Output the result ONLY as a compact JSON list of objects following this strict format :"
            # and strip everything up to the first '[', which should mark the start of the JSON array.
            
            # First, clean the prompt itself for comparison, removing spaces/newlines
            clean_prompt_end = "strict format :"
            
            # Try to find the start of the JSON by looking for the list bracket '['
            start_index = json_str.find('[')
            
            if start_index != -1:
                # Assuming the JSON starts at the first '['
                json_str = json_str[start_index:]
            else:
                # If '[' is not found, attempt to remove the prompt part
                if clean_prompt_end in json_str:
                     # This is a less reliable heuristic but might catch simple errors
                    json_str = json_str.split(clean_prompt_end)[-1].strip()


            # 3. Clean common VLM JSON wrappers (```json ... ```)
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0].strip()

            # 4. Check if the string is still empty or invalid before parsing
            if not json_str.startswith('['):
                return {"error": f"JSON parsing failed. Raw VLM output was not valid JSON after cleanup. Cleaned Output: {json_str}"}

            # Attempt to parse the cleaned JSON
            try:
                predictions = json.loads(json_str)
                
                # Validate, clean, and sort predictions
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
                # Return the detailed error including the cleaned output
                return {"error": f"JSON parsing failed after initial cleanup. Cleaned Output: {json_str}. Error: {e}"}

        st.success(f"VLM model ({VLM_MODEL_NAME}) loaded successfully!")
        return vlm_classifier_structured

    except Exception as e:
        st.error(f"Error loading VLM model: {e}")
        st.warning("Please check if the model name is correct and if you have enough VRAM for this VLM.")
        st.stop()
        return None

classifier = load_vlm_pipeline()

# --- Streamlit App Layout (Unchanged) ---
st.set_page_config(layout="wide", page_title="VLM Structured Classifier")
st.title("🧠 VLM Structured Classifier (Top 3 Scores)")

# --- 1. Load Data: User Image Upload (Unchanged) ---
st.header("1. Upload Your Images")
uploaded_files = st.file_uploader(
    "Choose one or more image files",
    type=['png', 'jpg', 'jpeg', 'webp'],
    accept_multiple_files=True
)

if 'selected_image_data' not in st.session_state:
    st.session_state.selected_image_data = None
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None
if 'last_uploaded_files' not in st.session_state:
    st.session_state.last_uploaded_files = []

if uploaded_files and (st.session_state.last_uploaded_files != uploaded_files):
    st.session_state.selected_image_data = None
    st.session_state.classification_result = None
    st.session_state.last_uploaded_files = uploaded_files

# --- 2. Display Images (Unchanged) ---
if uploaded_files:
    st.header("2. Select an Image for Classification")
    st.write("Click a button below an image to select it.")
    cols_per_row = 5
    image_cols = st.columns(cols_per_row)
    for i, uploaded_file in enumerate(uploaded_files):
        with image_cols[i % cols_per_row]:
            try:
                img_data = Image.open(uploaded_file)
                st.image(img_data, caption=f"{uploaded_file.name[:20]}...", width=150)
                if st.button(f"Select Image {i+1}", key=f"select_btn_{i}"):
                    st.session_state.selected_image_data = uploaded_file
                    st.session_state.classification_result = None
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

# --- 3. Classify Selected Image (Unchanged) ---
if st.session_state.selected_image_data:
    st.markdown("---")
    st.header("3. Classify Selected Image using Structured Prompt")
    st.write("The VLM is prompted to output a **JSON list** of the top 3 room types.")
    selected_file = st.session_state.selected_image_data
    st.write(f"**Selected Image:** `{selected_file.name}`")
    selected_img_to_display = Image.open(selected_file)
    st.image(selected_img_to_display, caption="Image to be classified", width=400)

    if st.button("🚀 Run VLM Classification (Top 3)"):
        if classifier is None:
            st.error("Model is not loaded. Cannot run classification.")
        else:
            with st.spinner("🧠 Generating structured predictions... Please wait."):
                try:
                    result = classifier(selected_img_to_display, prompt=FULL_VLM_PROMPT)
                    st.session_state.classification_result = result
                except Exception as e:
                    st.error(f"An unexpected error occurred during VLM classification: {e}")
                    st.session_state.classification_result = {"error": f"Classification failed ({e})"}
            st.rerun()

# --- 4. Show Result (Unchanged) ---
if st.session_state.classification_result:
    result = st.session_state.classification_result
    
    if isinstance(result, dict) and 'error' in result:
        st.error("Classification Error")
        st.code(result['error'])
    elif isinstance(result, list):
        st.success("✅ VLM Generation and Parsing Complete!")
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
            
        st.markdown("**Important Note:** These 'scores' are generated by the VLM's interpretation of probability based on the prompt, **not true logit probabilities** from a classification layer.")
    else:
        st.error("Received an unexpected result format from the VLM.")

if not uploaded_files:
    st.info("Please upload one or more images to begin.")
