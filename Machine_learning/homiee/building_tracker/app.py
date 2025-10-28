import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker, draw_tracked_objects

st.set_page_config(page_title="Building Tracker", layout="centered")

st.title("🏙️ Bird's-Eye Building Tracker")

st.markdown("""
Upload a short aerial video — this app detects and highlights buildings frame-by-frame
using a lightweight YOLOv8n model + Norfair tracker.
""")

uploaded_file = st.file_uploader("📁 Upload your video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)
    st.info("Processing video... please wait ⏳")

    # Load lightweight YOLO model
    model = YOLO("yolov8n.pt")

    # Norfair tracker
    tracker = Tracker(distance_function="euclidean", distance_threshold=30)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = os.path.join(tempfile.gettempdir(), "highlighted.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    progress = st.progress(0)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)
        dets = []
        for box in results[0].boxes:
            conf = float(box.conf[0])
            w, h = box.xywh[0][2:].cpu().numpy()
            # Only keep large detections (like buildings)
            if conf > 0.4 and w * h > 10000:
                dets.append(Detection(points=box.xywh[0][:2].cpu().numpy(), scores=conf))

        tracked_objects = tracker.update(detections=dets)
        draw_tracked_objects(frame, tracked_objects)

        overlay = frame.copy()
        for t in tracked_objects:
            x, y = map(int, t.estimate)
            cv2.circle(overlay, (x, y), 80, (0, 255, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        out.write(frame)
        frame_index += 1
        progress.progress(frame_index / total_frames)

    cap.release()
    out.release()

    st.success("✅ Done! Your highlighted video is ready below:")
    st.video(output_path)

    with open(output_path, "rb") as f:
        st.download_button(
            label="⬇️ Download highlighted video",
            data=f,
            file_name="highlighted_buildings.mp4",
            mime="video/mp4",
        )
