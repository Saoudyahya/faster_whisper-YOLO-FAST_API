# FastAPI Audio & Vision Processing API

A comprehensive FastAPI application that provides audio transcription and computer vision capabilities, including object detection, image captioning, and user guidance generation for visual navigation.

## Features

- **Audio Transcription**: Convert audio files to text using Faster-Whisper
- **Object Detection**: Detect and locate objects in images using YOLO
- **Image Captioning**: Generate descriptive captions for images
- **Visual Guidance**: Provide positional guidance for detected objects
- **Real-time Image Processing**: Fetch and process images from external URLs

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for better performance)
- Internet connection (for downloading models)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install fastapi uvicorn
   pip install faster-whisper
   pip install ultralytics
   pip install transformers
   pip install torch torchvision
   pip install opencv-python
   pip install pillow
   pip install requests
   ```

## Model Setup

The application will automatically download required models on first run:

- **YOLO Model**: `yolov9c.pt` (for object detection)
- **Whisper Model**: `tiny` model (for audio transcription)
- **Vision Transformer**: `nlpconnect/vit-gpt2-image-captioning` (for image captioning)

## Configuration

### Audio Processing
- Supported formats: WAV, MP3, M4A, FLAC
- Model: Faster-Whisper tiny (CPU optimized)
- Model storage: `./whisper_models/` directory

### Image Processing
- YOLO model: YOLOv9c for object detection
- Image captioning: ViT-GPT2 model
- External image source: Configurable URL endpoint

## API Endpoints

### 1. Audio Transcription

**Endpoint**: `POST /transcribe`

**Description**: Transcribe audio files to text with timestamps

**Request**:
```bash
curl -X POST "http://localhost:5000/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio.wav"
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "filename": "audio.wav",
    "duration": 15.32,
    "transcription": "Hello world, this is a test audio file.",
    "segments": [
      {
        "start": 0.0,
        "end": 3.5,
        "text": "Hello world,"
      },
      {
        "start": 3.5,
        "end": 7.2,
        "text": "this is a test audio file."
      }
    ]
  },
  "timestamp": "2025-01-15 10:30:45"
}
```

### 2. Image Processing

**Endpoint**: `POST /process_images`

**Description**: Process images for object detection, captioning, and guidance generation

**Request**:
```bash
curl -X POST "http://localhost:5000/process_images" \
     -H "accept: application/json"
```

**Response**:
```json
{
  "caption": "a group of people walking down a street",
  "detected_objects": [
    {
      "label": "person",
      "confidence": 0.95,
      "bounding_box": [100, 150, 200, 400]
    },
    {
      "label": "car",
      "confidence": 0.87,
      "bounding_box": [300, 200, 500, 350]
    }
  ],
  "guidance": "To your left, there is a car. To your front, there are 2 persons."
}
```

## Usage Examples

### Starting the Server

```bash
python test.py
```

The server will start on `http://localhost:5000`

### Python Client Example

```python
import requests

# Audio transcription
with open('audio.wav', 'rb') as audio_file:
    response = requests.post(
        'http://localhost:5000/transcribe',
        files={'file': audio_file}
    )
    print(response.json())

# Image processing
response = requests.post('http://localhost:5000/process_images')
print(response.json())
```

### JavaScript/Frontend Example

```javascript
// Audio transcription
const formData = new FormData();
formData.append('file', audioFile);

fetch('http://localhost:5000/transcribe', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));

// Image processing
fetch('http://localhost:5000/process_images', {
    method: 'POST'
})
.then(response => response.json())
.then(data => console.log(data));
```

## Architecture

### Core Components

1. **ObjectDetection**: YOLO-based object detection with bounding box coordinates
2. **ImageCaptioning**: Vision Transformer for generating image descriptions
3. **UserGuidance**: Spatial positioning system for navigation assistance
4. **MainApp**: Orchestrates all vision processing components

### Guidance System

The guidance system provides spatial information using a three-zone approach:
- **Left**: Objects in the left third of the image
- **Front**: Objects in the center third of the image
- **Right**: Objects in the right third of the image

## Configuration Options

### Changing Image Source

Update the `image_url` in the `/process_images` endpoint:

```python
image_url = "your-image-endpoint-url"
```

### Model Customization

```python
# Use different YOLO model
main_app = MainApp(
    yolo_path="yolov8n.pt",  # Lighter model
    caption_model_name="nlpconnect/vit-gpt2-image-captioning"
)

# Use different Whisper model
whisper_model = WhisperModel("base", device="cpu")  # More accurate
```

## Performance Optimization

### For CPU Usage
- Use smaller models (tiny Whisper, YOLOv8n)
- Reduce image resolution before processing
- Implement caching for repeated requests

### For GPU Usage
```python
# Enable GPU acceleration
whisper_model = WhisperModel("base", device="cuda")
```

## Error Handling

The API includes comprehensive error handling for:
- Invalid file formats
- Network connectivity issues
- Model loading failures
- Image processing errors

## Security Considerations

- File type validation for audio uploads
- CORS middleware configured for cross-origin requests
- Input sanitization and error boundaries

## Logging and Monitoring

Console logging is implemented for:
- Model loading status
- Processing progress
- Error tracking
- Performance metrics

## Troubleshooting

### Common Issues

1. **Model Download Failures**
   - Ensure stable internet connection
   - Check disk space for model storage
   - Verify firewall/proxy settings

2. **Audio Processing Errors**
   - Verify supported file formats
   - Check audio file integrity
   - Ensure sufficient memory

3. **Image Processing Issues**
   - Verify external image URL accessibility
   - Check image format compatibility
   - Monitor network timeout settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

