import os
import time
import streamlit as st
from ultralytics import YOLO
import cv2

# Fungsi untuk memproses video
def process_video(video_path, output_path):
    model = YOLO('/Users/owwl/Downloads/object_counting/best5.pt')
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error: Gagal membuka video input. Periksa path file.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.time()

    progress_bar = st.progress(0)
    status_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)

        elapsed_time = time.time() - start_time
        estimated_time = elapsed_time / progress if progress > 0 else 0
        status_text.text(f"Progress: {progress * 100:.2f}%")

    cap.release()
    out.release()
    st.success(f"Output video disimpan di: {output_path}")

# Tampilan Streamlit
st.title("YOLOv8 Video Processing with Streamlit")

uploaded_files = st.file_uploader("Upload video files", type=["mp4"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"Processing {uploaded_file.name}...")
        video_path = os.path.join("/tmp", uploaded_file.name)
        output_path = os.path.join("/tmp", f"processed_{uploaded_file.name}")

        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        process_video(video_path, output_path)

        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name=f"processed_{uploaded_file.name}",
                mime="video/mp4"
            )