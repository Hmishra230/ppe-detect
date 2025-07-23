// Main JavaScript for PPE Detection System

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initializeApp();
});

function initializeApp() {
    // Set up event listeners
    setupEventListeners();
    
    // Show initial status
    updateStatus('Ready to process. Select an input type and upload your file.', 'info');
}

function setupEventListeners() {
    // Input type change handler
    document.getElementById('inputType').addEventListener('change', function() {
        const selectedType = this.value;
        showInputSection(selectedType);
    });
}

function showInputSection(type) {
    // Hide all input sections
    const sections = document.querySelectorAll('.input-section');
    sections.forEach(section => {
        section.style.display = 'none';
    });
    
    // Show the selected section
    const targetSection = document.getElementById(type + 'Section');
    if (targetSection) {
        targetSection.style.display = 'block';
    }
    
    // Clear output container
    clearOutputContainer();
    
    // Update status
    updateStatus(`Selected ${type.toUpperCase()} input. Ready to process.`, 'info');
}

function updateStatus(message, type = 'info') {
    const statusElement = document.getElementById('statusMessage');
    const spinnerElement = document.getElementById('loadingSpinner');
    
    // Remove existing classes
    statusElement.className = 'alert';
    
    // Add appropriate class
    switch(type) {
        case 'success':
            statusElement.classList.add('alert-success', 'status-success');
            break;
        case 'error':
            statusElement.classList.add('alert-danger', 'status-error');
            break;
        case 'warning':
            statusElement.classList.add('alert-warning');
            break;
        default:
            statusElement.classList.add('alert-info');
    }
    
    statusElement.textContent = message;
    
    // Show/hide spinner
    if (type === 'loading') {
        statusElement.style.display = 'none';
        spinnerElement.style.display = 'block';
    } else {
        statusElement.style.display = 'block';
        spinnerElement.style.display = 'none';
    }
}

function clearOutputContainer() {
    const container = document.getElementById('outputContainer');
    container.innerHTML = `
        <div class="placeholder-content">
            <i class="fas fa-image fa-4x text-muted mb-3"></i>
            <p class="text-muted">Results will appear here after processing</p>
        </div>
    `;
}

function showLoadingInOutput() {
    const container = document.getElementById('outputContainer');
    container.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-3">Processing your file...</p>
        </div>
    `;
}

// Image Processing Functions
async function processImage() {
    const fileInput = document.getElementById('imageFile');
    const file = fileInput.files[0];
    
    if (!file) {
        updateStatus('Please select an image file first.', 'error');
        return;
    }
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        updateStatus('Please select a valid image file (JPG, PNG, etc.).', 'error');
        return;
    }
    
    // Show loading state
    updateStatus('Processing image...', 'loading');
    showLoadingInOutput();
    
    try {
        const formData = new FormData();
        formData.append('image', file);
        
        const response = await fetch('/process_image', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: formData
        });
        
        // Check if response is JSON
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (response.ok) {
            displayImageResult(result.image);
            updateStatus('Image processed successfully!', 'success');
        } else {
            throw new Error(result.error || 'Failed to process image');
        }
    } catch (error) {
        console.error('Error processing image:', error);
        updateStatus(`Error: ${error.message}`, 'error');
        clearOutputContainer();
    }
}

function displayImageResult(imageBase64) {
    const container = document.getElementById('outputContainer');
    container.innerHTML = `
        <img src="data:image/jpeg;base64,${imageBase64}" 
             alt="Processed Image" 
             class="img-fluid"
             style="max-width: 100%; max-height: 600px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
    `;
}

// Video Processing Functions
async function processVideo() {
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files[0];
    if (!file) {
        updateStatus('Please select a video file first.', 'error');
        return;
    }
    const formData = new FormData();
    formData.append('video', file);
    document.getElementById('status').innerHTML =
        '<div class="alert alert-info">Uploading video...</div>';
    try {
        const response = await fetch('/process_video', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (data.status === 'success') {
            // Debug log
            console.log('process_video response:', data);
            // Prefer stream_url, fallback to filename
            if (data.stream_url) {
                window.location.href = data.stream_url;
            } else if (data.filename) {
                window.location.href = '/video_stream/' + data.filename;
            } else {
                document.getElementById('status').innerHTML =
                    '<div class="alert alert-danger">Error: No stream URL returned from server.</div>';
            }
        } else {
            document.getElementById('status').innerHTML =
                '<div class="alert alert-danger">Error: ' + data.message + '</div>';
        }
    } catch (error) {
        document.getElementById('status').innerHTML =
            '<div class="alert alert-danger">Error: ' + error.message + '</div>';
    }
}

// CCTV Processing Functions
async function processCCTV() {
    const urlInput = document.getElementById('cctvUrl');
    const cctvUrl = urlInput.value.trim();
    
    if (!cctvUrl) {
        updateStatus('Please enter a CCTV feed URL.', 'error');
        return;
    }
    
    // Validate URL format
    try {
        new URL(cctvUrl);
    } catch (error) {
        updateStatus('Please enter a valid URL.', 'error');
        return;
    }
    
    // Show loading state
    updateStatus('Connecting to CCTV feed...', 'loading');
    showLoadingInOutput();
    
    try {
        const response = await fetch('/process_cctv', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: JSON.stringify({ cctv_url: cctvUrl })
        });
        
        if (response.ok) {
            updateStatus('CCTV feed connected successfully!', 'success');
            displayCCTVStream(cctvUrl);
        } else {
            // Check if response is JSON
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            throw new Error(result.error || 'Failed to connect to CCTV feed');
        }
    } catch (error) {
        console.error('Error processing CCTV:', error);
        updateStatus(`Error: ${error.message}`, 'error');
        clearOutputContainer();
    }
}

function displayCCTVStream(cctvUrl) {
    const container = document.getElementById('outputContainer');
    container.innerHTML = `
        <div class="text-center">
            <h5 class="mb-3">CCTV Feed</h5>
            <img src="/process_cctv" 
                 alt="CCTV Stream" 
                 class="video-stream"
                 style="max-width: 100%; max-height: 600px;">
            <p class="mt-3 text-muted">Live CCTV feed from: ${cctvUrl}</p>
        </div>
    `;
}

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function validateFileSize(file, maxSizeMB = 16) {
    const maxSizeBytes = maxSizeMB * 1024 * 1024;
    if (file.size > maxSizeBytes) {
        updateStatus(`File too large. Maximum size is ${maxSizeMB}MB.`, 'error');
        return false;
    }
    return true;
}

// Error handling
window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
    updateStatus('An unexpected error occurred. Please try again.', 'error');
});

// Handle page unload to clean up resources
window.addEventListener('beforeunload', function() {
    // Clean up any ongoing processes if needed
    console.log('Cleaning up resources...');
}); 