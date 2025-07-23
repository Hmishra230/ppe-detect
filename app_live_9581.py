import streamlit as st
import cv2
import torch
from ultralytics import YOLO

# Ensure Torch classes path is clear (for compatibility in some environments)
torch.classes.__path__ = []

# Load the trained YOLOv8 model
model_path = r"D:\INTERNSHIP25\PPE(HELMET)_final_1.6\NoneBoilerModel.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path).to(device)

# Thresholds
NMS_IOU_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.5

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        st.error(f"Error: Unable to load image at {image_path}")
        return
    results = model(image, iou=NMS_IOU_THRESHOLD)
    draw_boxes(image, results)
    st.image(image, channels="BGR")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    if not cap.isOpened():
        st.error(f"Error: Unable to open video at {video_path}")
        return

    frame_display = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, iou=NMS_IOU_THRESHOLD)
        draw_boxes(frame, results)
        frame_display.image(frame, channels="BGR")

    cap.release()

def process_cctv_feed(cctv_url):
    cap = cv2.VideoCapture(cctv_url)

    if not cap.isOpened():
        st.error(f"Error: Unable to open CCTV feed at {cctv_url}")
        return

    frame_display = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, iou=NMS_IOU_THRESHOLD)
        draw_boxes(frame, results)
        frame_display.image(frame, channels="BGR")

    cap.release()

def draw_boxes(image, results):
    """Draw red bounding boxes only for NO_Helmet and No_Boiler"""
    height, width, _ = image.shape

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, cls = box[:6]

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
                continue

            cls_int = int(cls)
            label = model.names[cls_int]

            # Only draw red box for NO_Helmet (2) and No_Boiler (3)
            if cls_int not in [2, 3]:
                continue

            color = (0, 0, 255)  # Red color

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(image, text, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Streamlit UI
st.title("PPE Detection System - Only 'No Helmet' & 'No Boiler'")

input_type = st.selectbox("Select input type", ["Image", "Video", "CCTV Feed"])

if input_type == "Image":
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        image_path = f"temp_{image_file.name}"
        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())
        process_image(image_path)

elif input_type == "Video":
    video_file = st.file_uploader("Upload a video", type=["mp4"])
    if video_file is not None:
        video_path = f"temp_{video_file.name}"
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        process_video(video_path)

elif input_type == "CCTV Feed":
    cctv_url = st.text_input("Enter CCTV feed URL")
    if cctv_url:
        process_cctv_feed(cctv_url)
