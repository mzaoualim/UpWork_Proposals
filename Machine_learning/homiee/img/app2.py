import streamlit as st
from PIL import Image
import torch
import json # Re-import json for parsing the VLM output

# Import the Hugging Face pipeline components for VLMs
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline

# --- Configuration ---
# Note: LLaVA-1.5-7B is used as a powerful example. 
# Other VLMs may require different prompt templates or cleanup.
VLM_MODEL_NAME = "llava-hf/llava-1.5-7b-hf" 

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
# Use a system-like instruction for reliability
FULL_VLM_PROMPT = f"USER: <image>\n{VLM_PROMPT_TEMPLATE}ASSISTANT:"

# --- Model Loading ---
@st.cache_resource
def load_vlm_pipeline():
    """Loads a Vision-Language Model (VLM) for structured output generation."""
    try:
        device = 0 if torch.cuda.is_available() else -1
        device_str = "GPU" if device == 0 else "CPU"
        st.info(f"Loading VLM model: {VLM_MODEL_NAME} on device: {device_str}...")

        # Load LLaVA components
        processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            VLM_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto" if device == 0 else "cpu",
        )
        
        def llava_classifier_structured(image, prompt):
            """Runs LLaVA generation and attempts to clean the output."""
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
            
            # Allow enough tokens for the JSON output (e.g., 200 tokens)
            output = model.generate(**inputs, max_new_tokens=200, temperature=0.0) 
            
            # Decode and clean the raw output
            response = processor.decode(output[0], skip_special_tokens=True)
            
            # Isolate the ASSISTANT's response
            answer_start = response.rfind("ASSISTANT:")
            if answer_start != -1:
                 generated_text = response[answer_start + len("ASSISTANT:"):].strip()
            else:
                 generated_text = response.strip()

            # Attempt to find and parse the JSON part of the generated text
            try:
                # Often the JSON is wrapped in ```json ... ``` blocks
                if '```json' in generated_text:
                    json_str = generated_text.split('```json')[1].split('```')[0].strip()
                elif '```' in generated_text:
                    # In case it uses a single block
                    json_str = generated_text.split('```')[1].split('```')[0].strip()
                else:
                    # Assume the whole response is the JSON array (best case)
                    json_str = generated_text
                
                # Load the JSON list
                predictions = json.loads(json_str)
                
                # Check for required fields and sort (VLM may not follow the prompt order)
                if isinstance(predictions, list):
                    predictions = [p for p in predictions if 'label' in p and 'score' in p]
                    # Sort by score descending and return only the top 3
                    return sorted(predictions, key=lambda x: x['score'], reverse=True)[:3]
                else:
                    raise ValueError("VLM response was not a JSON list.")

            except Exception as e:
                # Return the raw text if parsing fails for debugging
                return {"error": f"JSON parsing failed. Raw VLM output: {generated_text}. Error: {e}"}

        st.success(f"VLM model ({VLM_MODEL_NAME}) loaded successfully!")
        return llava_classifier_structured

    except Exception as e:
        st.error(f"Error loading VLM model: {e}")
        st.warning("Ensure you have sufficient VRAM and that the model supports multimodal generation.")
        st.stop()
        return None

classifier = load_vlm_pipeline()

# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="VLM Structured Classifier")
st.title("🧠 VLM Structured Classifier (Top 3 Scores)")

# --- 1. Load Data: User Image Upload (Unchanged) ---
st.header("1. Upload Your Images")
uploaded_files = st.file_uploader(
    "Choose one or more image files",
    type=['png', 'jpg', 'jpeg', 'webp'],
    accept_multiple_files=True
)

# Initialize session state variables (Unchanged)
if 'selected_image_data' not in st.session_state:
    st.session_state.selected_image_data = None
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None
if 'last_uploaded_files' not in st.session_state:
    st.session_state.last_uploaded_files = []

# Clear selection logic (Unchanged)
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

# --- 3. Classify Selected Image (Modified) ---
if st.session_state.selected_image_data:
    st.markdown("---")
    st.header("3. Classify Selected Image using Structured Prompt")
    st.write("The VLM is prompted to output a JSON list of the top 3 room types.")

    selected_file = st.session_state.selected_image_data
    selected_img_to_display = Image.open(selected_file)
    st.image(selected_img_to_display, caption="Image to be classified", width=400)

    if st.button("🚀 Run VLM Classification (Top 3)"):
        if classifier is None:
            st.error("Model is not loaded. Cannot run classification.")
        else:
            with st.spinner("🧠 Generating structured predictions... Please wait."):
                try:
                    # Result is now a list of dictionaries (top 3) or an error dictionary
                    result = classifier(selected_img_to_display, prompt=FULL_VLM_PROMPT)
                    st.session_state.classification_result = result
                    
                except Exception as e:
                    st.error(f"An unexpected error occurred during VLM classification: {e}")
                    st.session_state.classification_result = {"error": f"Classification failed ({e})"}
            st.rerun()

# --- 4. Show Result (Modified) ---
if st.session_state.classification_result:
    result = st.session_state.classification_result
    
    # Check if the result is an error dictionary
    if isinstance(result, dict) and 'error' in result:
        st.error(result['error'])
        
    # Check if the result is a successful list of predictions
    elif isinstance(result, list):
        st.success("✅ VLM Generation and Parsing Complete!")
        st.subheader("Top 3 Predicted Room Types")
        
        # Display each of the top predictions
        for i, prediction in enumerate(result):
            label = prediction.get('label', 'N/A')
            score = prediction.get('score', 0.0)
            
            # Ensure score is a number between 0 and 1 for the progress bar
            try:
                score = float(score)
            except ValueError:
                score = 0.0
            
            st.markdown(f"**#{i+1}: {label}**")
            st.progress(score, text=f"Confidence: {score:.2%}")
            
        st.markdown("**Important Note:** These 'scores' are generated by the VLM's interpretation of probability based on the prompt, not true logit probabilities from a classification layer.")
        
    else:
        st.error("Received an unexpected result format from the VLM.")

if not uploaded_files:
    st.info("Please upload one or more images to begin.")
