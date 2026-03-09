
import os

if not os.path.exists("best3.pt"):
    gdown.download("https://drive.google.com/uc?id=1npBPlOmoHYuvoCSaoiGx8SJOCR9Pf9rv", "best3.pt", quiet=False)

import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import pandas as pd

# Load YOLOv8 model
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Run prediction and filter by confidence
def predict_image(model, image, conf_threshold):
    results = model(image)
    filtered = []
    for result in results:
        boxes = []
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                boxes.append((x1, y1, x2, y2, label, conf))
        filtered.append(boxes)
    return filtered

# Draw only highest confidence bounding box
def draw_results(image, predictions):
    if not predictions or not predictions[0]:
        return image, None
    best_box = max(predictions[0], key=lambda x: x[5])
    x1, y1, x2, y2, label, conf = best_box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image, [best_box]

# Process video frame-by-frame
def process_video(model, video_path, conf_threshold):
    cap = cv2.VideoCapture(video_path)
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(temp_video.name, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        predictions = predict_image(model, frame, conf_threshold)
        frame, _ = draw_results(frame, predictions)
        out.write(frame)

    cap.release()
    out.release()
    return temp_video.name

# Streamlit UI setup
st.set_page_config(page_title="🚀 YOLOv8 Detection", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #1f1c2c, #928dab);
            color: white;
        }
        .title {
            font-size: 3rem;
            font-weight: bold;
            color: #FF6347;
            text-align: center;
            margin-bottom: 10px;
        }
        .upload-box {
            border: 2px dashed #FF5733;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            background-color: rgba(255, 255, 255, 0.05);
            margin-top: 20px;
        }
        .stButton>button {
            background: linear-gradient(90deg, #FF5733, #FF8D33);
            border: none;
            color: white;
            font-size: 16px;
            padding: 8px 20px;
            border-radius: 8px;
        }
        .stSlider > div {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🚀 YOLOv8 Object Detection (Top Confidence Only)</div>', unsafe_allow_html=True)
st.write("Upload an image or video to detect the object with the highest confidence using YOLOv8.")

# Controls
option = st.radio("Choose input type", ["Image", "Video"], horizontal=True)
conf_threshold = st.slider("🔧 Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Load YOLO model
model_path = "best3.pt"  # Replace with your own model file
model = load_model(model_path)

# Upload section
uploaded_file = st.file_uploader("📤 Upload File (Image or Video)", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1]
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    if option == "Image" and file_extension in ["jpg", "jpeg", "png"]:
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        predictions = predict_image(model, image, conf_threshold)
        output_image, best = draw_results(image.copy(), predictions)

        # Display side-by-side images
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, channels="BGR", caption="🖼️ Original Image", use_container_width=True)

        with col2:
            st.image(output_image, channels="BGR", caption="🔍 Detected Image (Top Confidence)", use_container_width=True)

        # Display detection details
        st.markdown("### 🏆 Detection Details")
        if best:
            df = pd.DataFrame(best, columns=["X1", "Y1", "X2", "Y2", "Label", "Confidence"])
            df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.2f}")
            st.dataframe(df)
        else:
            st.warning("No objects detected with the selected confidence threshold.")

    elif option == "Video" and file_extension == "mp4":
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

        st.video(temp_file_path)
        with st.spinner("⏳ Processing video..."):
            processed_video_path = process_video(model, temp_file_path, conf_threshold)
        st.video(processed_video_path, caption="✅ Processed Video Output")
        os.remove(temp_file_path)

st.success("✅ Ready! Upload your image or video above.")
