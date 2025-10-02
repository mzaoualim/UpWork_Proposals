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
    prompt = "<image>\nWhat type of room is shown in this image? Respond in 1-3 words only."

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

    # -------------------
    # Cleanup postprocessing
    # -------------------
    answer = raw_answer.strip()

    # Remove the original prompt if echoed
    if "What type of room" in answer:
        answer = answer.split("\n")[-1]

    # Drop multiple-choice noise (e.g., "A. Bedroom. B. Bedroom...")
    answer = answer.split("A.")[0] if "A." in answer else answer

    # Keep only the first sentence/line
    answer = answer.split("\n")[0].split(".")[0]

    # Limit to 3 words max
    answer = " ".join(answer.split()[:3])

    return answer.strip().title()


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
