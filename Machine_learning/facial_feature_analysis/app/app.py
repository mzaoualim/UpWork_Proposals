import streamlit as st
import requests
import mediapipe as mp
import opencv-python as cv2
import cvlib as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

## Domain Knowledge Table for Jaw Classification
range_dict = {
    'Male':
    {'Strong': (0.45, 0.55),
    'Average': (0.40, 0.50),
    'Weak': (0.35, 0.45)},
    'Female':
    {'Strong': (0.40, 0.50),
    'Average': (0.35, 0.45),
    'Weak': (0.30, 0.35)}
}
## Download the model file
url = "https://huggingface.co/spaces/VictorPanther/SSI_Gender_Detection/resolve/main/gender_detection.h5"
response = requests.get(url, stream=True)
response.raise_for_status()  # Raise an exception for bad responses

with open("gender_detection.h5", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

gender_model = load_model('/content/gender_detection.h5')
      
## Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect_and_crop_face(image):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face = image[y:y+h, x:x+w]
                return cv2.resize(face, (224, 224))  # Standardize size
    return None

def classify_gender(face_image):
    # Resize the image to the expected input shape of the model (96, 96)
    face_image = cv2.resize(face_image, (96, 96))  
    face_image = face_image / 255.0  # Normalize
    face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension
    prediction = gender_model.predict(face_image)
    return 'Male' if prediction[0][0] > 0.5 else 'Female'

def jaw_ratio(face_image):
    # Use MediaPipe to get facial landmarks
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Example: Calculate jaw length and face length using landmarks
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
    return None, None

## Classify Jaw Strength
def classify_jaw_strength(gender, jaw_ratio):
    '''
    Jaw length / face length ratio: This ratio is calculated by dividing the jaw length by the face length. A higher ratio typically indicates a stronger jaw.
    '''
    for strength, (lower, upper) in range_dict[gender].items():
        if lower <= jaw_ratio <= upper:
            return strength
    return 'Unable to determine the Jaw Strength \n The computed Jaw ratio is located out of the domaine knowledge range\n'

def process_image(image_path):
    image = cv2.imread(image_path)
    face = detect_and_crop_face(image)
    if face is not None:
        gender = classify_gender(face)
        calculated_jaw_ratio = jaw_ratio(face)
        jaw_strength = classify_jaw_strength(gender, calculated_jaw_ratio)
        return gender, jaw_strength
    return None, None

def main():
    st.markdown("<h1 style='text-align: center;'> PREDICT AND PROVIDE INSIGHTS ON LOAN APPROVAL DECISIONS </h1>", unsafe_allow_html=True)

if __name__ == '__main__':
  main()