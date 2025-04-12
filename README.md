# Crop-Weed Detection System

![Project Preview](https://via.placeholder.com/800x400.png?text=Project+Demo) 
*Add actual screenshots of your interface here*

A deep learning-based web application for detecting crops and weeds in agricultural images using YOLOv5 and FastAPI.

## Features

- 🖼️ Image upload functionality for analysis
- 🎯 Real-time object detection with YOLOv5 model
- 🔍 Bounding box visualization with confidence scores
- 📊 Detailed detection results table
- 📱 Responsive web interface
- 🚀 FastAPI backend for efficient processing

### Technologies Used

- **Backend:** FastAPI, Python

- **ML Framework:** PyTorch, YOLOv5

- **Computer Vision:** PIL, OpenCV (through YOLO)

- **Frontend:** HTML5, CSS3, Jinja2 templating

- **Deployment:** Render (via render.yaml)

## Getting Started 

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/crop-weed-detection.git
cd crop-weed-detection
```
### 2.Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3.Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### 1.Start the FastAPI server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Access the web interface at:
```bash
http://localhost:8000
```
Upload an image through the web interface to see detection results

Note: The model (~27MB) will be automatically downloaded on first run from Hugging Face.

## Project Structure
```bash
crop-weed-detection/
├── templates/           # HTML templates
│   └── index.html
├── static/             # Static assets
│   ├── uploads/        # User-uploaded images
│   └── results/        # Processed images with detections
├── main.py             # FastAPI application
├── requirements.txt    # Python dependencies
├── render.yaml         # Deployment configuration
└── crop_weed_detection_model.pt  # Pretrained YOLOv5 model
```

### License

[License](https://license/)

## Acknowledgments

- YOLOv5 implementation by Ultralytics

- Model hosting on Hugging Face

- FastAPI documentation and community

- **Contributors:** [umeshsamartapu]()
- **Maintainer:** [umeshsamartapu]() [MeruguBharath]()
- **Status:** Active development

## Demo 
### You can watch the ([youtube video](    )) for demo
<p align="center">
  <img src="  " />
</p>  


## 📫 Let's Connect

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/umesh-samartapu-42793025a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
[![Twitter](https://img.shields.io/badge/-Twitter-1DA1F2?style=flat-square&logo=twitter&logoColor=white)](https://x.com/umeshsamartapu?t=graUTdTs4QlUc3a5OOH7hA&s=09)
[![Email](https://img.shields.io/badge/-Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:umeshsamartapu@gmail.com)
[![Instagram](https://img.shields.io/badge/-Instagram-E4405F?style=flat-square&logo=instagram&logoColor=white)](https://www.instagram.com/umeshsamartapu?igsh=MWsxbWVzbHd0bDgyag==)
[![Buy Me a Coffee](https://img.shields.io/badge/-Buy%20Me%20a%20Coffee-FBAD19?style=flat-square&logo=buymeacoffee&logoColor=black)](https://www.buymeacoffee.com/umeshsamartapu)

---

🔥 Always exploring new technologies and solving real-world problems with code!
