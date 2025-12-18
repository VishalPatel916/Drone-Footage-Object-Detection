import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import time

st.set_page_config(page_title="Drone Object Detection", layout="wide")
st.title("🚁 Drone Object Detection Dashboard")

model = YOLO("models/best.pt")

# Sidebar options
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Image Upload", "Video Upload", "Live Webcam"]
)

# ================= IMAGE =================
if mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(uploaded_file.read())
        img = cv2.imread(temp.name)

        results = model(img)
        annotated = results[0].plot()
        st.image(annotated, channels="BGR")

# ================= VIDEO =================
elif mode == "Video Upload":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4"])
    if uploaded_file:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(uploaded_file.read())
        st.video(temp.name)

        if st.button("Run Detection"):
            model.predict(source=temp.name, save=True)
            st.success("Detection saved in runs/detect/predict")

# ================= WEBCAM =================
elif mode == "Live Webcam":
    st.warning("Press STOP to end webcam stream")

    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break

            results = model(frame, conf=0.4)
            annotated_frame = results[0].plot()

            FRAME_WINDOW.image(annotated_frame, channels="BGR")
            time.sleep(0.03)  # ~30 FPS

        cap.release()
