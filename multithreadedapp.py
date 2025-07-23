import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import threading
import queue
import time

# Load YOLOv8 model
model_path = "NoneBoilerModel.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path).to(device)

# Thresholds
NMS_IOU_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.5

def draw_boxes(image, results):
    """Draw red bounding boxes only for NO_Helmet"""
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

            if cls_int != 2:  # Only No_Helmet
                continue

            color = (0, 0, 255)  # Red box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(image, text, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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

    stframe = st.empty()
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, iou=NMS_IOU_THRESHOLD)
        draw_boxes(frame, results)

        # FPS overlay
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()

def process_cctv_feed(cctv_url):
    cap = cv2.VideoCapture(cctv_url)
    if not cap.isOpened():
        st.error(f"Error: Unable to open CCTV feed at {cctv_url}")
        return

    stframe = st.empty()
    frame_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    # Thread: Capture frames and add to queue
    def capture_frames():
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                stop_event.set()
                break
            if not frame_queue.full():
                frame_queue.put(frame)
            time.sleep(0.01)

    t1 = threading.Thread(target=capture_frames)
    t1.start()

    frame_id = 0
    prev_time = time.time()

    try:
        # MAIN THREAD: Read from queue, process, and render
        while not stop_event.is_set():
            if not frame_queue.empty():
                frame = frame_queue.get()
                frame_id += 1

                if frame_id % 2 != 0:
                    continue  # skip odd frames

                # Inference and overlay
                start_time = time.time()
                results = model(frame, iou=NMS_IOU_THRESHOLD)
                draw_boxes(frame, results)

                fps = 1.0 / (start_time - prev_time)
                prev_time = start_time
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # ✅ Only now update Streamlit UI
                stframe.image(frame, channels="BGR")
            else:
                time.sleep(0.01)
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        stop_event.set()
        t1.join()
        cap.release()


# ----------------------------
# Streamlit UI
# ----------------------------

st.title("🛡️ PPE Detection - Only 'No Helmet'")
input_type = st.selectbox("Select input type", ["Image", "Video", "CCTV Feed"])

if input_type == "Image":
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        image_path = f"temp_{image_file.name}"
        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())
        process_image(image_path)

elif input_type == "Video":
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if video_file is not None:
        video_path = f"temp_{video_file.name}"
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        process_video(video_path)

elif input_type == "CCTV Feed":
    cctv_url = st.text_input("Enter RTSP/CCTV feed URL")
    if cctv_url:
        process_cctv_feed(cctv_url)
