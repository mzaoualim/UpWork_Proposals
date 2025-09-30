import streamlit as st
from PIL import Image
import torch
import io

# Hugging Face imports
from transformers import AutoProcessor, AutoModelForVision2Seq

# --- Configuration ---
QWEN_MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"


VLM_PROMPT_TEMPLATE = """
You are an expert at identifying rooms. 
Look at the image and answer: What type of room is this? 
Respond concisely in 2-4 words only.
Examples: "Living room", "Bathroom", "Hotel lobby", "Garage".
"""

# --- Model Loading ---
@st.cache_resource
def load_qwen_vl_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_map = "auto" if device.type == "cuda" else "cpu"
        st.info(f"Loading {QWEN_MODEL_NAME} on {device} (map: {device_map})...")

        processor = AutoProcessor.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=(
                torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            ),
            device_map=device_map,
            trust_remote_code=True
        )

        st.success(f"{QWEN_MODEL_NAME} loaded successfully on {device}!")
        return processor, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
        return None, None, None

processor, model, device = load_qwen_vl_model()

# --- Room Type Inference ---
def predict_room_type(image: Image.Image):
    prompt = "What type of room is shown in this image? Respond in 1-3 words only."
    inputs = processor(
        text=[prompt],
        images=[image.convert("RGB")],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=0.0
        )

    answer = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    if "```" in answer:
        answer = answer.split("```")[-1].strip()
    return answer


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Qwen2.5-VL Room Type Classifier")
st.title("🧠 Qwen2.5-VL Room Type Classifier (No Predefined Labels)")

# --- 1. Upload Data ---
st.header("1. Upload Your Images")
uploaded_files = st.file_uploader(
    "Choose one or more image files",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)

# Session state
if "selected_image_data" not in st.session_state:
    st.session_state.selected_image_data = None
if "classification_result" not in st.session_state:
    st.session_state.classification_result = None
if "last_uploaded_files" not in st.session_state:
    st.session_state.last_uploaded_files = []

# Reset selection if files changed
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
                img_data = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
                uploaded_file.seek(0)
                st.image(img_data, caption=f"{uploaded_file.name[:20]}...", width=150)
                if st.button(f"Select Image {i+1}", key=f"select_btn_{i}"):
                    st.session_state.selected_image_data = uploaded_file
                    st.session_state.classification_result = None
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

# --- 3. Run Classification ---
if st.session_state.selected_image_data:
    st.markdown("---")
    st.header("3. Classify Selected Image")
    st.write(f"Model: **{QWEN_MODEL_NAME}** running on **{device}**")

    selected_file = st.session_state.selected_image_data
    st.write(f"**Selected Image:** `{selected_file.name}`")

    selected_file.seek(0)
    selected_img = Image.open(io.BytesIO(selected_file.read()))
    st.image(selected_img, caption="Image to be classified", width=400)

    if st.button("🚀 Run Room Type Inference"):
        if model is None:
            st.error("Model not loaded.")
        else:
            with st.spinner("🔎 Inferring room type..."):
                try:
                    result = infer_room_type(selected_img)
                    st.session_state.classification_result = result
                except Exception as e:
                    st.session_state.classification_result = f"❌ Error: {e}"
            st.rerun()

# --- 4. Show Results ---
if st.session_state.classification_result:
    result = st.session_state.classification_result
    st.success("✅ Inference Complete!")
    st.subheader("Predicted Room Type")
    st.markdown(f"**{result}**")

if not uploaded_files:
    st.info("Please upload one or more images to begin.")
