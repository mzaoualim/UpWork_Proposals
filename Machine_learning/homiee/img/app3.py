import streamlit as st
from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    CLIPProcessor,
    CLIPModel
)

# ------------------
# Config
# ------------------
VLM_MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"  # fallback: SmolVLM-256M-Instruct
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------
# Load Models
# ------------------
@st.cache_resource
def load_vlm():
    try:
        processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            VLM_MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        return processor, model
    except Exception as e:
        st.error(f"❌ Error loading VLM: {e}")
        return None, None

@st.cache_resource
def load_clip():
    try:
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        return clip_processor, clip_model
    except Exception as e:
        st.error(f"❌ Error loading CLIP: {e}")
        return None, None

vlm_processor, vlm_model = load_vlm()
clip_processor, clip_model = load_clip()

# ------------------
# Prediction Functions
# ------------------
def predict_with_vlm(image: Image.Image) -> str:
    if vlm_processor is None or vlm_model is None:
        return "Model not loaded"

    prompt = (
        "<image>\n"
        "Identify the type of room shown in this image. "
        "Answer with only the room type. Provide options A, B, C with different plausible answers."
    )

    inputs = vlm_processor(
        text=[prompt],
        images=[image.convert("RGB")],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        out = vlm_model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.0
        )

    raw_answer = vlm_processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    # --- Post-processing ---
    # Extract "A.", "B.", "C." parts only
    matches = re.findall(r"(A\..*?)(?=B\.|$)|"
                         r"(B\..*?)(?=C\.|$)|"
                         r"(C\..*?)(?=D\.|$)", raw_answer, re.DOTALL)

    # Flatten and clean
    choices = [m.strip() for tup in matches for m in tup if m]

    # Fallback if nothing matched
    if not choices:
        return "Unknown"

    return " ".join(choices)

def predict_with_clip(image: Image.Image, candidate_labels: list) -> str:
    if clip_processor is None or clip_model is None:
        return "Model not loaded"

    inputs = clip_processor(
        text=candidate_labels,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        logits_per_image = clip_model(**inputs).logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    best_idx = probs.argmax()
    return f"{candidate_labels[best_idx]} ({probs[best_idx]:.2f})"

# ------------------
# Streamlit UI
# ------------------
st.set_page_config(layout="wide", page_title="SmolVLM + CLIP Room Classifier")
st.title("🏠 Room Type Classifier with SmolVLM + CLIP")

uploaded_file = st.file_uploader(
    "Upload an image of a room",
    type=["jpg", "jpeg", "png", "webp"]
)

labels_input = st.text_input(
    "Enter candidate labels for CLIP (comma-separated):",
    value="Bedroom, Living Room, Kitchen, Bathroom, Dining Room, Office"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Room Image", width=400)

    if st.button("🔎 Classify with Both Models"):
        with st.spinner("Analyzing with SmolVLM..."):
            vlm_result = predict_with_vlm(image)

        with st.spinner("Analyzing with CLIP..."):
            candidate_labels = [lbl.strip() for lbl in labels_input.split(",") if lbl.strip()]
            clip_result = predict_with_clip(image, candidate_labels)

        st.subheader("📊 Results")
        st.success(f"🧠 **SmolVLM Prediction:** {vlm_result}")
        st.success(f"🎯 **CLIP Prediction:** {clip_result}")
else:
    st.info("Please upload a room image to begin.")
