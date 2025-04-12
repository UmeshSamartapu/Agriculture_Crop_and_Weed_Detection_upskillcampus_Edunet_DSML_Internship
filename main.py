import os
import requests
import torch
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageFont
from werkzeug.utils import secure_filename

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuration
MODEL_PATH = 'crop_weed_detection_model.pt'
MODEL_URL = "https://huggingface.co/UmeshSamartapu/NewProject/resolve/main/crop_weed_detection_model.pt"
UPLOAD_DIR = "static/uploads"
RESULT_DIR = "static/results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Model downloaded successfully")

def load_model():
    download_model()
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
        model.conf = 0.5  # Confidence threshold
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

async def detect_image(img_path):
    if model is None:
        return []
    
    results = model(img_path)
    return results.pandas().xyxy[0].to_dict(orient='records')

def draw_boxes(image_path, detections):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for detection in detections:
        xmin = detection['xmin']
        ymin = detection['ymin']
        xmax = detection['xmax']
        ymax = detection['ymax']
        label = detection['name']
        confidence = detection['confidence']

        # Draw bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=2)
        
        # Create text background
        text = f"{label} {confidence:.2f}"
        text_bbox = draw.textbbox((xmin, ymin), text, font=font)
        draw.rectangle(text_bbox, fill='red')
        
        # Draw text
        draw.text((xmin, ymin), text, fill='white', font=font)
    
    result_path = os.path.join(RESULT_DIR, os.path.basename(image_path))
    img.save(result_path)
    return result_path

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_DIR, filename)
        
        with open(upload_path, 'wb') as f:
            f.write(contents)
        
        detections = await detect_image(upload_path)
        result_path = draw_boxes(upload_path, detections)
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "uploaded_image": filename,
            "result_image": os.path.basename(result_path),
            "detections": detections
        })
    
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)