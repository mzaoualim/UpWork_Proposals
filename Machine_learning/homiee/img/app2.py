import streamlit as st
from PIL import Image
import torch
import io
import spacy
import json

from transformers import AutoProcessor, AutoModelForCausalLM

# --- Configuration ---
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

CAPTION_PROMPT = "Describe the contents of this image in one clear sentence."

# --- Load Spacy NLP ---
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# --- Load Qwen2.5-VL ---
@st.cache_resource
def load_qwen_vl_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_map = "auto" if device.type == "cuda" else "cpu"
        st.info(f"Loading Qwen2.5-VL model ({QWEN_MODEL_NAME}) on {device}...")

        processor = AutoProcessor.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            device_map=device_map,
            trust_remote_code=True
        )

        st.success("✅ Model loaded successfully!")
        return processor, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

processor, model, device = load_qwen_vl_model()

# --- Caption Function ---
def generate_caption(image: Image.Image, prompt: str = CAPTION_PROMPT) -> str:
    inputs = processor(
        text=prompt,
        images=image.convert("RGB"),
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.0
        )

    caption = processor.decode(out[0], skip_special_tokens=True).strip()
    return caption

# --- Inference ---
def infer_room_type_open(image: Image.Image):
    # Ask Qwen directly for room type
    prompt = "What type of room is shown in this image? Respond in a few words."
    inputs = processor(
        text=prompt,
        images=image.convert("RGB"),
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=0.0
        )

    answer = processor.decode(out[0], skip_special_tokens=True).strip()
    return answer

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Open-Vocab Room Detector (Qwen2.5-VL)")
st.title("🏠 Open-Vocabulary Room Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    if st.button("🔎 Detect Room Type"):
        with st.spinner("Generating caption with Qwen2.5-VL..."):
            caption = generate_caption(img)

        result = infer_room_type(caption)

        st.markdown(f"**Caption:** {result['caption']}")
        st.markdown(f"**Inferred Room Type:** {result['room_type']}")
        st.markdown(f"**Reasoning:** {result['reasoning']}")
