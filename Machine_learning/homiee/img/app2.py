import streamlit as st
from PIL import Image
import torch
import json
from transformers import pipeline, AutoProcessor, AutoModelForCausalLM # Import components

# --- Configuration ---
# Using a model known for VQA/Image-to-Text tasks that generally works well 
# with the 'image-to-text' pipeline. (BLIP-VQA is used for stability).
VLM_MODEL_NAME = "Salesforce/blip-vqa-base" 
# VLM_MODEL_NAME = "llava-hf/llava-1.5-7b-hf" # You can try LLaVA again, but it's much larger.

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
# Use a simple prompt structure suitable for the 'image-to-text' pipeline
FULL_VLM_PROMPT = f"{VLM_PROMPT_TEMPLATE}"

# --- Model Loading (FIXED) ---
@st.cache_resource
def load_vlm_pipeline():
    """
    Loads a Vision-Language Model (VLM) for structured output generation 
    using the dedicated 'image-to-text' pipeline (FIXED to avoid AutoModelForCausalLM error).
    """
    try:
        device = 0 if torch.cuda.is_available() else -1
        device_str = "GPU" if device == 0 else "CPU"
        st.info(f"Loading VLM model: {VLM_MODEL_NAME} on device: {device_str}...")

        # 🎯 FIX: Use the 'image-to-text' pipeline for multimodal models.
        model_pipeline = pipeline(
            "image-to-text",
            model=VLM_MODEL_NAME,
            device=device 
        )
        
        # --- Wrapper Function ---
        def vlm_classifier_structured(image, prompt):
            """Runs VLM generation and attempts to clean the structured JSON output."""
            
            # The 'image-to-text' pipeline uses the 'prompt' argument for VQA/VLM questions
            # Setting low temperature for deterministic (classification-like) output
            result = model_pipeline(
                image, 
                prompt=prompt, 
                max_new_tokens=200, 
                do_sample=False, 
                temperature=0.0
            )
            generated_text = result[0]['generated_text']

            # Attempt to find and parse the JSON part of the generated text
            try:
                json_str = generated_text.strip()
                
                # Common cleanup steps for VLM JSON output
                if '```json' in json_str:
                    json_str = json_str.split('```json')[1].split('```')[0].strip()
                elif '```' in json_str:
                    json_str = json_str.split('```')[1].split('```')[0].strip()
                
                # Load the JSON list
                predictions = json.loads(json_str)
                
                # Validate, clean, and sort predictions
                if isinstance(predictions, list):
                    # Filter out entries missing essential keys and convert score to float
                    def get_score(p):
                        try:
                            return float(p.get('score', 0.0))
                        except ValueError:
                            return 0.0
                            
                    predictions = [p for p in predictions if 'label' in p and 'score' in p]
                    
                    # Sort by score descending and return only the top 3
                    return sorted(predictions, key=get_score, reverse=True)[:3]
                else:
                    raise ValueError("VLM response was not a valid JSON list.")

            except Exception as e:
                # Return the raw text if parsing fails for debugging
                return {"error": f"JSON parsing failed. Raw VLM output: {generated_text}. Error: {e}"}

        st.success(f"VLM model ({VLM_MODEL_NAME}) loaded successfully!")
        return vlm_classifier_structured

    except Exception as e:
        st.error(f"Error loading VLM model: {e}")
        st.warning("Please check if the model name is correct and if you have enough VRAM for this VLM.")
        st.stop()
        return None

classifier = load_vlm_pipeline()

# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="VLM Structured Classifier")
st.title("🧠 VLM Structured Classifier (Top 3 Scores)")

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
                img_data = Image.open(uploaded_file)
                st.image(img_data, caption=f"{uploaded_file.name[:20]}...", width=150)

                if st.button(f"Select Image {i+1}", key=f"select_btn_{i}"):
                    st.session_state.selected_image_data = uploaded_file
                    st.session_state.classification_result = None
                    st.rerun()

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

# --- 3. Classify Selected Image ---
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

# --- 4. Show Result ---
if st.session_state.classification_result:
    result = st.session_state.classification_result
    
    # Check if the result is an error dictionary
    if isinstance(result, dict) and 'error' in result:
        st.error("Classification Error")
        st.code(result['error'])
        
    # Check if the result is a successful list of predictions
    elif isinstance(result, list):
        st.success("✅ VLM Generation and Parsing Complete!")
        st.subheader("Top 3 Predicted Room Types")
        
        # Display each of the top predictions
        for i, prediction in enumerate(result):
            label = prediction.get('label', 'N/A')
            score = prediction.get('score', 0.0)
            
            # Ensure score is a float
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
