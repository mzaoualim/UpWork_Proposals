import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# --- Configuration ---
MODEL_NAME = "HuggingFaceTB/SmolVLM-Instruct"

# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_map = "auto" if device.type == "cuda" else "cpu"

        st.info(f"Loading {MODEL_NAME} on {device} (map: {device_map})...")

        processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            trust_remote_code=True
        )

        st.success("✅ Model loaded successfully!")
        return processor, model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

processor, model, device = load_model()

# --- Room Prediction Function ---
def predict_room_type(image: Image.Image):
    # Include <image> token in the prompt to match #images passed
    prompt = "<image>\nWhat type of room is shown in this image? Respond in 1-3 words only."

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

    # Clean markdown fences if present
    if "```" in answer:
        answer = answer.split("```")[-1].strip()

    return answer

# --- Streamlit Layout ---
st.set_page_config(layout="wide", page_title="SmolVLM Room Classifier")
st.title("🏠 SmolVLM Room Classifier")

# 1. Upload Images
st.header("1. Upload Your Images")
uploaded_files = st.file_uploader(
    "Choose image files",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    st.header("2. Select an Image")
    cols = st.columns(5)
    for i, uploaded_file in enumerate(uploaded_files):
        with cols[i % 5]:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption=uploaded_file.name, width=150)
            if st.button(f"Select {uploaded_file.name}", key=f"btn_{i}"):
                st.session_state.selected_image = img
                st.session_state.selected_name = uploaded_file.name
                st.session_state.result = None
                st.rerun()

# 2. Classification
if "selected_image" in st.session_state:
    st.markdown("---")
    st.header("3. Classification")
    st.write(f"**Selected Image:** `{st.session_state.selected_name}`")
    st.image(st.session_state.selected_image, caption="Image to classify", width=400)

    if st.button("🚀 Run Classification"):
        with st.spinner("Analyzing with SmolVLM..."):
            try:
                result = predict_room_type(st.session_state.selected_image)
                st.session_state.result = result
            except Exception as e:
                st.error(f"❌ Error during classification: {e}")
                st.session_state.result = None
        st.rerun()

# 3. Results
if "result" in st.session_state and st.session_state.result:
    st.success("✅ Classification Complete!")
    st.subheader("Predicted Room Type")
    st.markdown(f"**{st.session_state.result}**")

if not uploaded_files:
    st.info("Upload one or more images to start.")
