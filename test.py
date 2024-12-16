from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from faster_whisper import WhisperModel
from ultralytics import YOLO
from transformers import (
    AutoTokenizer,
    VisionEncoderDecoderModel,
    ViTImageProcessor
)
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict
from collections import defaultdict, Counter
import os
import io
import torch
import requests
import uvicorn

class ObjectDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_objects(self, image_path):
        img = cv2.imread(image_path)
        results = self.model(img)
        detected_objects = []
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result
            label = self.model.names[int(cls)]
            detected_objects.append({
                "label": label,
                "confidence": round(conf, 2),
                "bounding_box": [int(x1), int(y1), int(x2), int(y2)]
            })
        return detected_objects

class ImageCaptioning:
    def __init__(self, model_name):
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_caption(self, image):
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values,
                max_length=50,
                num_beams=4,
                early_stopping=True
            )

        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption

class UserGuidance:
    @staticmethod
    def generate_guidance(detected_objects: List[Dict], caption: str = "", image_width: int = 640) -> str:
        if not detected_objects:
            return "No objects detected."

        position_groups = {"left": [], "front": [], "right": []}
        for obj in detected_objects:
            x1, y1, x2, y2 = obj["bounding_box"]
            label = obj["label"]
            center_x = (x1 + x2) / 2
            
            # First-person perspective positioning
            # Swapping left and right to match camera view
            if center_x < image_width / 3:
                position_groups["right"].append(label)
            elif center_x > 2 * image_width / 3:
                position_groups["left"].append(label)
            else:
                position_groups["front"].append(label)

        # Convert the guidance to a string directly
        guidance_parts = []
        for position in ["left", "front", "right"]:
            if position_groups[position]:
                counter = Counter(position_groups[position])
                descriptions = []
                for obj, count in counter.most_common():
                    if count > 1:
                        descriptions.append(f"{count} {obj}s")
                    else:
                        descriptions.append(f"a {obj}")

                if descriptions:
                    # Improved joining of descriptions
                    if len(descriptions) > 1:
                        objects_text = ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"
                    else:
                        objects_text = descriptions[0]
                    
                    guidance_parts.append(f"To your {position}, there is {objects_text}.")

        return " ".join(guidance_parts) if guidance_parts else "No distinct objects detected in specific positions."

    @staticmethod
    def merge_guidance(all_guidance: List[str], captions: List[str]) -> str:
        # If captions exist, prepend them to the first guidance
        if captions and all_guidance:
            return f"Scene overview: {' '.join(captions)} {all_guidance[0]}"
        elif captions:
            return f"Scene overview: {' '.join(captions)}"
        elif all_guidance:
            return all_guidance[0]
        else:
            return "No scene description available."

class MainApp:
    def __init__(self, yolo_path, caption_model_name):
        self.detector = ObjectDetection(yolo_path)
        self.captioner = ImageCaptioning(caption_model_name)
        self.guidance_generator = UserGuidance()

    def process_single_image(self, image_path):
        detected_objects = self.detector.detect_objects(image_path)
        image = cv2.imread(image_path)
        image_caption = self.captioner.generate_caption(image)
        guidance = self.guidance_generator.generate_guidance(detected_objects)
        
        return {
            "caption": image_caption,
            "detected_objects": detected_objects,
            "guidance": guidance
        }

# Initialize the MainApp with model paths
main_app = MainApp(
    yolo_path="yolov9c.pt",
    caption_model_name="nlpconnect/vit-gpt2-image-captioning"
)

# Specify the directory for the Whisper model
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'whisper_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Load the Faster-Whisper model
# Check if model files already exist
model_path = os.path.join(MODEL_DIR, "tiny")
if not os.path.exists(model_path):
    print("Downloading model...")
    whisper_model = WhisperModel("tiny", device="cpu", download_root=MODEL_DIR)
else:
    print("Loading pre-downloaded model...")
    whisper_model = WhisperModel(model_path, device="cpu")

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    allowed_extensions = {'wav', 'mp3', 'm4a', 'flac'}
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Check file extension
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    try:
        # Read file content
        audio_bytes = await file.read()
        
        # Print processing message
        print(f"\nProcessing audio file: {file.filename}")
        print("-" * 50)
        
        # Transcribe the audio file directly from memory
        segments, info = whisper_model.transcribe(io.BytesIO(audio_bytes))
        
        # Extract and print the transcribed text
        transcribed_text = ""
        print("\nTranscription:")
        print("=" * 50)
        transcription_segments = []
        for segment in segments:
            transcribed_text += f"{segment.text} "
            # Print each segment with timestamp
            print(f"[{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text}")
            
            # Prepare segments for JSON response
            transcription_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text
            })
        
        print("=" * 50)
        print(f"Total duration: {info.duration:.2f} seconds")
        print("-" * 50 + "\n")
        
        # Format the response
        response_data = {
            'filename': file.filename,
            'duration': info.duration,
            'transcription': transcribed_text.strip(),
            'segments': transcription_segments
        }
        
        return JSONResponse(content={
            'status': 'success',
            'data': response_data,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        # Print error message
        print(f"\nError during transcription: {str(e)}")
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_images")
async def process_images():
    image_url = "https://190c-105-67-6-147.ngrok-free.app/latest_frame"
    
    try:
        # Fetch the image from the URL
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Read the image content
        img_data = io.BytesIO(response.content)

        # Save the image temporarily to process
        img_path = 'temp_image.jpg'
        with open(img_path, 'wb') as temp_file:
            temp_file.write(img_data.getvalue())

        # Process the image
        result = main_app.process_single_image(img_path)
        result_dict = {
            'caption': result['caption'],
            'detected_objects': result['detected_objects'],
            'guidance': result['guidance']
        }

        # Clean up temporary image file
        if os.path.exists(img_path):
            os.remove(img_path)

        return result_dict

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching image from URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)