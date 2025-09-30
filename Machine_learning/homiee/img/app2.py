import streamlit as st
from PIL import Image
import torch torchvision
import json
import io 

# Import the specific Hugging Face classes for VLM with Causal LM backbone
from transformers import AutoProcessor, AutoModelForCausalLM

# --- Configuration ---
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
# Significantly smaller (~3.6B parameters) VLM that excels at instruction following.

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
# Qwen models often use a simpler formatting than LLaVA, but using the general "image + text" input works well.
FULL_VLM_PROMPT = VLM_PROMPT_TEMPLATE

# --- Model Loading (FIXED for Qwen2.5-VL) ---
@st.cache_resource
def load_qwen_vl_model():
    """
    Loads the Qwen2.5-VL processor and model.
    """
    try:
        # Determine the device and device map
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # For smaller models, "auto" or directly 'cuda' is fine if VRAM permits (approx 8-10GB for this 3.6B model)
        device_map = "auto" if device.type == "cuda" else "cpu"
        st.info(f"Loading Qwen2.5-VL model: {QWEN_MODEL_NAME} on device: {device} (Map: {device_map})...")

        # Use AutoProcessor and AutoModelForCausalLM, as it's a Causal LM backbone
        processor = AutoProcessor.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            device_map=device_map,
            trust_remote_code=True
        )
        
        st.success(f"Qwen2.5-VL model loaded successfully on {device}!")
        return processor, model, device

    except Exception as e:
        st.error(f"Error loading Qwen2.5-VL model: {e}")
        st.warning("Ensure 'accelerate' and 'transformers' are installed, and check VRAM.")
        st.stop()
        return None, None, None

processor, model, device = load_qwen_vl_model()

# --- Qwen Classification Function ---
def vlm_classifier_structured(raw_image: Image.Image, prompt: str):
    """
    Performs Qwen classification and reliably parses the structured JSON output.
    """
    if model is None:
        return {"error": "Model is not loaded."}
        
    # Qwen processor uses a dedicated way to create model inputs from image and text
    inputs = processor(
        text=prompt, 
        images=raw_image.convert('RGB'),
        return_tensors="pt"
    ).to(device)

    # Generate the answer text
    out = model.generate(
        **inputs, 
        max_new_tokens=400, 
        do_sample=False, 
        temperature=0.0
    )
    
    # Decode the raw output
    generated_text = processor.decode(out[0], skip_special_tokens=True).strip()

    # --- JSON Parsing Logic (Standardized for robustness) ---
    json_str = generated_text
    
    # 1. Look for and clean common VLM JSON wrappers
    if '```json' in json_str:
        # Use partitioning to safely extract the content between the code block markers
        json_str = json_str.partition('```json')[2].partition('```')[0].strip()
    elif '```' in json_str:
        json_str = json_str.partition('```')[2].partition('```')[0].strip()

    # 2. Look for the start of the JSON array
    start_index = json_str.find('[')
    if start_index != -1:
        json_str = json_str[start_index:]
    
    # 3. Simple JSON validation check (must start with '[')
    if not json_str.startswith('['):
        return {"error": f"JSON parsing failed. Qwen did not generate a JSON list. Cleaned Output: {json_str}"}

    # 4. Attempt to parse the cleaned JSON
    try:
        predictions = json.loads(json_str)
        
        if isinstance(predictions, list):
            # Helper to safely convert score to float
            def get_score(p):
                try:
                    return float(p.get('score', 0.0))
                except (ValueError, TypeError):
                    return 0.0
                    
            predictions = [p for p in predictions if 'label' in p and 'score' in p]
            
            # Sort and return the top 3
            return sorted(predictions, key=get_score, reverse=True)[:3]
        else:
            raise ValueError("VLM response was not a valid JSON list.")

    except Exception as e:
        return {"error": f"JSON parsing failed after cleanup. Cleaned Output: {json_str}. Error: {e}"}


# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Qwen2.5-VL Structured Classifier")
st.title("🧠 Qwen2.5-VL Structured Classifier (Top 3 Scores via JSON)")

# --- 1. Load Data: User Image Upload ---
st.header("1. Upload Your Images")
uploaded_files = st.file_uploader(
    "Choose one or more image files",
    type=['png', 'jpg', 'jpeg', 'webp'],
    accept_multiple_files=True
)

# Initialize session state variables (omitted for brevity, assume they exist)
if 'selected_image_data' not in st.session_state: st.session_state.selected_image_data = None
if 'classification_result' not in st.session_state: st.session_state.classification_result = None
if 'last_uploaded_files' not in st.session_state: st.session_state.last_uploaded_files = []

# Clear selection logic
if uploaded_files and (st.session_state.last_uploaded_files != uploaded_files):
    st.session_state.selected_image_data = None
    st.session_state.classification_result = None
    st.session_state.last_uploaded_files = uploaded_files

# --- 2. Display Images ---
if uploaded_files:
    st.header("2. Select an Image for Classification")
    cols_per_row = 5
    image_cols = st.columns(cols_per_row)
    for i, uploaded_file in enumerate(uploaded_files):
        with image_cols[i % cols_per_row]:
            try:
                img_data = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')
                uploaded_file.seek(0)
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
    st.write(f"Model: **{QWEN_MODEL_NAME}** (Small and efficient!) running on **{device}**")
    
    selected_file = st.session_state.selected_image_data
    st.write(f"**Selected Image:** `{selected_file.name}`")
    
    selected_file.seek(0)
    selected_img_to_display = Image.open(io.BytesIO(selected_file.read()))
    st.image(selected_img_to_display, caption="Image to be classified", width=400)

    if st.button("🚀 Run Qwen2.5-VL Classification (Top 3)"):
        if model is None:
            st.error("Model is not loaded. Cannot run classification.")
        else:
            with st.spinner("🧠 Generating structured predictions..."):
                try:
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
        st.error("Classification Error (Failed JSON Generation)")
        st.code(result['error'])
    elif isinstance(result, list):
        st.success("✅ Qwen2.5-VL Generation and Parsing Complete!")
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
    else:
        st.error("Received an unexpected result format from the VLM.")

if not uploaded_files:
    st.info("Please upload one or more images to begin.")
