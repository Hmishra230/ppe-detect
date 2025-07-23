import os
import cv2
import torch
import base64
import tempfile
import uuid
from ultralytics import YOLO
from flask import Flask, render_template, request, jsonify, Response, send_file, abort
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import io
import logging
import time

# Ensure Torch classes path is clear (for compatibility in some environments)
torch.classes.__path__ = []

# --- Logging ---
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store active video processing sessions
active_videos = {}

# Load the trained YOLOv8 model
model_path = r"F:\INTERNSHIP_June25\INTERNSHIP25\PPE(HELMET)_final_1.6\NoneBoilerModel.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path).to(device)

# Thresholds
NMS_IOU_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.5

# Error handlers to ensure JSON responses
@app.errorhandler(400)
def bad_request(error):
    if request.headers.get('Accept', '').startswith('application/json'):
        return jsonify({'error': 'Bad request', 'message': str(error)}), 400
    return render_template('error.html', error=error), 400

@app.errorhandler(404)
def not_found_error(error):
    if 'application/json' in request.headers.get('Accept', ''):
        return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404
    return render_template('error.html', error=error), 404

@app.errorhandler(405)
def method_not_allowed(error):
    if request.headers.get('Accept', '').startswith('application/json'):
        return jsonify({'error': 'Method not allowed', 'message': 'The HTTP method is not allowed for this endpoint'}), 405
    return render_template('error.html', error=error), 405

@app.errorhandler(413)
def too_large(error):
    if 'application/json' in request.headers.get('Accept', ''):
        return jsonify({'status': 'error', 'message': 'File too large'}), 413
    return render_template('error.html', error=error), 413

@app.errorhandler(500)
def internal_error(error):
    if 'application/json' in request.headers.get('Accept', ''):
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500
    return render_template('error.html', error=error), 500

@app.errorhandler(Exception)
def handle_exception(error):
    if request.headers.get('Accept', '').startswith('application/json'):
        return jsonify({'error': 'Server error', 'message': str(error)}), 500
    return render_template('error.html', error=error), 500

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

def process_image(image_path):
    """Process image and return base64 encoded result"""
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Unable to load image at {image_path}")
    
    results = model(image, iou=NMS_IOU_THRESHOLD)
    draw_boxes(image, results)
    
    # Convert to base64 for web display
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64, None

def process_video_frame(frame):
    """Process a single video frame"""
    results = model(frame, iou=NMS_IOU_THRESHOLD)
    draw_boxes(frame, results)
    return frame

def generate_video_frames(video_path):
    """Generate processed video frames for streaming"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    if not cap.isOpened():
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n'
               b'Error: Unable to open video file\r\n\r\n')
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_video_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        # Clean up the video file
        try:
            os.remove(video_path)
        except:
            pass

def generate_cctv_frames(cctv_url):
    """Generate processed CCTV frames for streaming"""
    cap = cv2.VideoCapture(cctv_url)

    if not cap.isOpened():
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n'
               b'Error: Unable to connect to CCTV feed\r\n\r\n')
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_video_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

def generate_frames(video_path):
    import cv2
    import sys
    import os
    print(f"[DEBUG] Attempting to stream video: {video_path}")
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file does not exist: {video_path}")
        # Yield a placeholder error image
        error_img = get_error_image_bytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + error_img + b'\r\n')
        return
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video file: {video_path}")
        error_img = get_error_image_bytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + error_img + b'\r\n')
        return
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"[DEBUG] No more frames to read or failed to read frame at count {frame_count}.")
            break
        results = model(frame, iou=NMS_IOU_THRESHOLD)
        draw_boxes(frame, results)
        ret2, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        frame_count += 1
    print(f"[DEBUG] Streaming finished. Total frames: {frame_count}")
    cap.release()

def get_error_image_bytes():
    # Return a simple red X image as JPEG bytes
    import numpy as np
    import cv2
    img = 255 * np.ones((200, 400, 3), dtype=np.uint8)
    cv2.putText(img, 'Video Error', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
    ret, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image_route():
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image file provided'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        # Save to temp file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            file.save(temp.name)
            temp_path = temp.name
        # Process image
        result_path = process_image(temp_path)
        return jsonify({
            'status': 'success',
            'message': 'Image processed successfully',
            'result_url': f'/get_result/{os.path.basename(result_path)}'
        })
    except Exception as e:
        app.logger.exception("Image processing failed")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_result/<filename>')
def get_result(filename):
    # Serve processed image
    result_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(result_path):
        abort(404)
    return send_file(result_path, mimetype='image/jpeg')

@app.route('/process_video', methods=['POST'])
def process_video_route():
    try:
        if 'video' not in request.files:
            return jsonify({'status': 'error', 'message': 'No video file provided'}), 400
        file = request.files['video']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        # Save to temp_uploads
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        temp_filename = f"video_{int(time.time())}_{file.filename}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_path)
        return jsonify({
            'status': 'success',
            'message': 'Video uploaded successfully',
            'stream_url': f'/video_stream/{temp_filename}',
            'filename': temp_filename
        })
    except Exception as e:
        app.logger.exception("Video processing failed")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/process_cctv', methods=['POST'])
def process_cctv_route():
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'status': 'error', 'message': 'No CCTV URL provided'}), 400
        cctv_url = data['url']
        # Your existing CCTV processing logic here
        return jsonify({
            'status': 'success',
            'message': 'CCTV feed processing started'
        })
    except Exception as e:
        app.logger.exception("CCTV processing failed")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/video_stream/<filename>')
def video_stream_page(filename):
    return render_template('video_stream.html', filename=filename)

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"[DEBUG] /video_feed called with: {video_path}")
    return Response(generate_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 