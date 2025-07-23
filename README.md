# PPE Detection System - Flask Web Application

A Flask web application for Personal Protective Equipment (PPE) detection using YOLOv8. This application detects 'No Helmet' and 'No Boiler' violations in images, videos, and CCTV feeds.

## Features

- **Image Processing**: Upload and process images for PPE detection
- **Video Processing**: Upload and stream processed videos in real-time
- **CCTV Feed Processing**: Connect to CCTV feeds for live monitoring
- **Real-time Detection**: Only shows red bounding boxes for violations (No Helmet and No Boiler)
- **Responsive Web Interface**: Modern, mobile-friendly UI with Bootstrap

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- Web browser with JavaScript enabled

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the YOLO model file is present**:
   - The model file `NoneBoilerModel.pt` should be in the project directory
   - Update the model path in `app.py` if needed

## Usage

1. **Start the Flask application**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Select input type**:
   - **Image**: Upload and process single images
   - **Video**: Upload and stream processed videos
   - **CCTV Feed**: Enter CCTV URL for live monitoring

4. **Upload files or enter CCTV URL** and click the process button

## File Structure

```
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── NoneBoilerModel.pt     # YOLO model file
├── templates/
│   └── index.html        # Main web interface
├── static/
│   ├── css/
│   │   └── style.css     # Custom styles
│   └── js/
│       └── main.js       # Frontend JavaScript
└── temp_uploads/         # Temporary file storage
```

## API Endpoints

- `GET /` - Main application page
- `POST /process_image` - Process uploaded images
- `POST /process_video` - Upload and process videos
- `GET /video_stream/<video_id>` - Stream processed video
- `POST /process_cctv` - Process CCTV feed

## Configuration

### Model Settings
- **NMS IOU Threshold**: 0.45
- **Confidence Threshold**: 0.5
- **Detection Classes**: Only No Helmet (class 2) and No Boiler (class 3) are highlighted

### File Upload Limits
- **Maximum file size**: 16MB
- **Supported image formats**: JPG, JPEG, PNG
- **Supported video formats**: MP4

## Troubleshooting

### Common Issues

1. **"Method Not Allowed" Error**:
   - Ensure you're using the latest version of the application
   - Check that all routes are properly configured

2. **Model Loading Error**:
   - Verify the model file path in `app.py`
   - Ensure the model file exists and is accessible

3. **Video Streaming Issues**:
   - Check browser console for JavaScript errors
   - Ensure popup blockers are disabled
   - Verify video file format is supported

4. **CCTV Connection Issues**:
   - Verify the CCTV URL is accessible
   - Check network connectivity
   - Ensure the CCTV feed supports the required format

### Performance Optimization

- **GPU Acceleration**: Install CUDA-compatible PyTorch for faster processing
- **Memory Management**: Large videos may require more RAM
- **Network**: CCTV feeds require stable internet connection

## Technical Details

### Detection Logic
- Uses YOLOv8 model for object detection
- Only displays red bounding boxes for violation classes
- Processes frames in real-time for video/CCTV feeds
- Automatic cleanup of temporary files

### Web Interface
- Responsive design using Bootstrap 5
- Real-time status updates
- File upload validation
- Error handling and user feedback

## License

This project is for educational and research purposes.

## Support

For issues or questions, please check the troubleshooting section above or review the application logs. 