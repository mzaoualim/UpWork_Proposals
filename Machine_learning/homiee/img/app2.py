import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import json
import re

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
        model = AutoModelForCausalLM.from_pretrained(
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
# Prediction Function
# ------------------
def predict_room_type(image: Image.Image):
    prompt = """<image>
Based on this image, list the TOP 3 most likely room types with confidence scores (between 0.0 and 1.0).
Respond ONLY with a compact JSON array, strictly in this format:
[
  { "label": "Living Room", "score": 0.92 },
  { "label": "Bedroom", "score": 0.05 },
  { "label": "Kitchen", "score": 0.03 }
]
Do not include explanations or extra text.
"""

    inputs = processor(
        text=[prompt],
        images=[image.convert("RGB")],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.0
        )

    raw_answer = processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    # --- Cleanup ---
    if "```json" in raw_answer:
        raw_answer = raw_answer.split("```json")[-1].split("```")[0].strip()
    elif "```" in raw_answer:
        raw_answer = raw_answer.split("```")[1].strip()

    start = raw_answer.find("[")
    end = raw_answer.rfind("]") + 1
    if start != -1 and end != -1:
        raw_answer = raw_answer[start:end]

    # --- Parse JSON ---
    try:
        predictions = json.loads(raw_answer)
        cleaned = []
        for p in predictions:
            label = str(p.get("label", "Unknown")).strip()
            try:
                score = float(p.get("score", 0.0))
            except Exception:
                score = 0.0
            cleaned.append({"label": label, "score": max(0.0, min(1.0, score))})

        return sorted(cleaned, key=lambda x: x["score"], reverse=True)[:3]

    except Exception:
        # Fallback → extract just 1 word
        match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Za-z]+){0,2})\b", raw_answer)
        if match:
            return [{"label": match.group(1), "score": 1.0}]
        else:
            return [{"label": "Unknown", "score": 1.0}]


# ------------------
# Streamlit UI
# ------------------
st.set_page_config(page_title="Room Type Classifier", layout="centered")
st.title("🏠 Room Type Classifier")
st.write("Upload an image and let SmolVLM guess the type of room!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file and processor and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Classify Room"):
        with st.spinner("Analyzing image..."):
            result = predict_room_type(image)
            st.session_state.result = result

# Results
if "result" in st.session_state and st.session_state.result:
    result = st.session_state.result
    if isinstance(result, list):
        st.success("✅ Classification Complete!")
        st.subheader("Top 3 Predicted Room Types")
        for i, pred in enumerate(result):
            st.markdown(f"**#{i+1}: {pred['label']}**")
            st.progress(pred["score"], text=f"Confidence: {pred['score']:.2%}")
    else:
        st.error("❌ Unexpected output format.")
