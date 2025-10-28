import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker, draw_tracked_objects

st.set_page_config(page_title="Building Highlighter", layout="centered")

st.title("🏙️ Bird's-Eye Building Highlighter")

st.markdown("""
Upload a short aerial video — this app detects and highlights buildings frame-by-frame
using a lightweight YOLOv8 model + Norfair tracker.
""")

uploaded_file = st.file_uploader("📁 Upload your video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    st.info("Processing video... please wait ⏳")

    # Load YOLO model (lightest version)
    model = YOLO("yolov8n.pt")

    tracker = Tracker(distance_function="euclidean", distance_threshold=30)

    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(tempfile.gettempdir(), "highlighted.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    progress = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model.predict(frame, verbose=False)
        dets = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            # Filter for building-like large objects if YOLO detects many things
            if conf > 0.4 and box.xywh[0][2] * box.xywh[0][3] > 10000:  
                dets.append(Detection(points=box.xywh[0].cpu().numpy()[:2], scores=conf))

        tracked_objects = tracker.update(detections=dets)
        draw_tracked_objects(frame, tracked_objects)

        # Semi-transparent overlay
        overlay = frame.copy()
        for t in tracked_objects:
            x, y = map(int, t.estimate)
            cv2.circle(overlay, (x, y), 80, (0, 255, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        out.write(frame)

        processed += 1
        progress.progress(min(processed / frame_count, 1.0))

    cap.release()
    out.release()

    st.success("✅ Done! Your highlighted video is ready:")
    st.video(output_path)

    st.download_button(
        label="⬇️ Download highlighted video",
        data=open(output_path, "rb").read(),
        file_name="highlighted_buildings.mp4",
        mime="video/mp4"
    )
