import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# ------------------
# Config
# ------------------
MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------
# Load model + processor
# ------------------
@st.cache_resource
def load_model():
    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        return processor, model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

processor, model = load_model()

# ------------------
# Prediction function
# ------------------
def predict_room_type(image: Image.Image) -> str:
    # Stronger, clearer prompt
    prompt = (
        "<image>\n"
        "Identify the type of room shown in this image. "
        "Answer with only the room type in 1–3 words (e.g., 'Bedroom', 'Living Room', 'Kitchen')."
    )

    inputs = processor(
        text=[prompt],
        images=[image.convert("RGB")],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=0.0
        )

    raw_answer = processor.batch_decode(out, skip_special_tokens=True)[0]
    answer = raw_answer.strip()

    # --- Post-processing cleanup ---
    # Remove echoed prompt text
    if "Identify the type of room" in answer or "What type of room" in answer:
        answer = answer.split("\n")[-1]

    # Remove multiple-choice noise
    if "A." in answer:
        answer = answer.split("A.")[0]

    # Keep only first sentence/line
    answer = answer.split("\n")[0].split(".")[0]

    # Limit to 3 words, Title Case
    answer = " ".join(answer.split()[:3]).title()

    # Safety net: discard junk like single characters
    if len(answer) < 3 or not any(c.isalpha() for c in answer):
        answer = "Unknown"

    return answer

# ------------------
# Streamlit UI
# ------------------
st.set_page_config(layout="wide", page_title="SmolVLM Room Classifier")
st.title("🏠 Room Type Classifier (SmolVLM-256M)")

uploaded_file = st.file_uploader(
    "Upload an image of a room",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Room Image", width=400)

    if st.button("🔎 Classify Room"):
        if processor is None or model is None:
            st.error("❌ Model is not loaded. Check logs.")
        else:
            with st.spinner("Analyzing room type..."):
                result = predict_room_type(image)
                st.success(f"🏷️ Predicted Room Type: **{result}**")
else:
    st.info("Please upload a room image to begin.")

