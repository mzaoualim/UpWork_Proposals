import streamlit as st
import requests
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import os

# Setup retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

# Model download with caching
@st.cache_resource
def download_model():
    model_path = "gender_detection.h5"
    
    # Check if model already exists
    if os.path.exists(model_path):
        return load_model(model_path)
    
    url = "https://huggingface.co/spaces/VictorPanther/SSI_Gender_Detection/resolve/main/gender_detection.h5"
    
    try:
        # Add headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = http.get(url, stream=True, headers=headers)
        response.raise_for_status()

        # Save the model
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return load_model(model_path)
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            st.error("Too many requests to the model server. Please try again later.")
        else:
            st.error(f"Error downloading model: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

# Load the model
try:
    gender_model = download_model()
    if gender_model is None:
        st.error("Failed to load the model. Please refresh the page or try again later.")
        st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# range_dict
range_dict = {}
      
## Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect_and_crop_face(image_array):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        results = face_detection.process(image_array)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image_array.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face = image_array[y:y+h, x:x+w]
                # Convert to PIL Image, resize, and back to numpy array
                face_pil = Image.fromarray(face)
                face_pil = face_pil.resize((224, 224))
                return np.array(face_pil)
    return None

def classify_gender(face_image):
    # Resize using PIL
    face_pil = Image.fromarray(face_image)
    face_pil = face_pil.resize((96, 96))
    face_image = np.array(face_pil)
    
    face_image = face_image / 255.0  # Normalize
    face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension
    prediction = gender_model.predict(face_image)
    return 'Male' if prediction[0][0] > 0.5 else 'Female'

def jaw_ratio(face_image):
    # Use MediaPipe to get facial landmarks
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(face_image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # jaw length
                jaw_points = [(landmarks[i].x, landmarks[i].y) for i in range(0, 17)]
                left_jaw = jaw_points[0]
                right_jaw = jaw_points[16]

                jaw_length = np.linalg.norm(np.array(left_jaw) - np.array(right_jaw))

                # face length
                left_eye = landmarks[10]
                right_eye = landmarks[338]
                chin = landmarks[152]

                ## Calculate midpoint of eyes
                eye_midpoint_x = (left_eye.x + right_eye.x) / 2
                eye_midpoint_y = (left_eye.y + right_eye.y) / 2

                ## Calculate face length
                face_length = np.linalg.norm(np.array([eye_midpoint_x, eye_midpoint_y]) - np.array([chin.x, chin.y]))

                return jaw_length / face_length
    return None

def process_image(image_file):
    # Read image using PIL
    image = Image.open(image_file)
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Convert to numpy array
    image_array = np.array(image)
    
    face = detect_and_crop_face(image_array)
    if face is not None:
        gender = classify_gender(face)
        calculated_jaw_ratio = jaw_ratio(face)
        return gender, calculated_jaw_ratio
    return None, None

def main():
    st.markdown("<h1 style='text-align: center;'> JAW STRENGTH CLASSIFIER </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'> STEP 1: UPLOAD IMAGE </h2>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader('Image Uploader', type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    
    if uploaded_image:
        gender, calculated_jaw_ratio = process_image(uploaded_image)
        a, b, c = st.columns(3)
        with b:
            st.image(uploaded_image)
        st.subheader(f'Predicted gender: {gender}')

        st.markdown("<h2 style='text-align: center;'> STEP 2: SELECT JAW STRENGTH RATIO RANGE </h2>", unsafe_allow_html=True)
        st.info('Based on your domain knowledge,\nSelect corresponding ranges for jaw length to face length ratio to express jaw strength.')
        weak_jaws = st.slider("weak jaws", 0.0, 1.0, (0.2, 0.4))
        medium_jaws = st.slider("medium jaws", 0.0, 1.0, (weak_jaws[1], 0.6))
        strong_jaws = st.slider("strong jaws", 0.0, 1.0, (medium_jaws[1], 0.7))

        st.markdown("<h2 style='text-align: center;'> STEP 3: FINAL RESULTS </h2>", unsafe_allow_html=True)
        
        generate_results = st.button('Generate Results!', use_container_width=True)
        if generate_results:
            range_dict = {
                'Weak': weak_jaws,
                'Average': medium_jaws,
                'Strong': strong_jaws,
            }

            in_range = False
            for strength, band in range_dict.items():
                if band[0] <= calculated_jaw_ratio <= band[1]:
                    st.write(f' The computed ratio equals : {np.round(calculated_jaw_ratio,2)}\n')
                    st.write(f'The captured jaws are considered {strength}')
                    in_range = True
            
            if not in_range:
                st.write(f' The computed ratio equals : {np.round(calculated_jaw_ratio,2)}\n')
                st.write('Unable to determine the Jaw Strength')
                st.write('The computed Jaw Strength ratio is located outside the domaine knowledge range')

if __name__ == '__main__':
    main()
