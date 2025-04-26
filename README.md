# 🌾 Crop and Weed Detection System

A complete deep learning-based solution for detecting crops and weeds in agricultural images, using YOLOv5, FastAPI, and PyTorch.

# 📁 Project Directory Structure
```bash
|_ Application/   # Web app for detection (FastAPI + YOLOv5)
|_ Code/          # Model training and evaluation scripts
|_ DataSet/       # Labeled dataset (images + annotations)
```

# 📌 Application - Crop-Weed Detection Web App

[WebApp]()

A web application that allows users to upload agricultural images and detects crops and weeds in real-time.

# 🔥 Features

- Image upload and detection

- Real-time YOLOv5 object detection

- Bounding box visualization

- FastAPI backend for efficient serving

- Responsive and simple frontend (HTML/CSS)

# 🚀 Quick Start

```bash
git clone https://github.com/UmeshSamartapu/Agriculture_Crop_and_Weed_Detection_upskillcampus_Edunet_DSML_Internship
cd crop-weed-detection/Application
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
Access the web app at: http://localhost:8000
```

The model (~27MB) will download automatically on first run!

# 📌 Code - Crop-Weed Detection Pipeline
A PyTorch-based training and inference pipeline for classifying crops and weeds from labeled images.

# ⚙️ Highlights

- Uses YOLO-formatted dataset for training

- Backbone: ResNet-18 pretrained

- Visualization of bounding boxes

- GPU (CUDA) support

- Model checkpoint saving (best_model.pt)

# 🛠️ Requirements
```bash
pip install torch torchvision opencv-python scikit-learn matplotlib seaborn pandas Pillow
```

# 🚀 To Run
```bash
python crop_weed_detection.py --data_dir path_to_dataset
```

# Outputs:
```bash
/models/best_model.pt

/sample_visualizations

/predictions
```

# 📌 DataSet - Crops and Weeds Images
This project uses a custom labeled dataset:

Download Dataset: [Google Drive Link]()

Dataset Structure:
```bash
|_ DataSet/
|    |_ *.img   # 1300 crop and weed images
|    |_ *.txt   # Corresponding YOLO-format annotations
|_ classes.txt  # Class list: [crop, weed]
```

# 🙌 Acknowledgments

- Ultralytics YOLOv5

- FastAPI Documentation

- Model hosted on Hugging Face

- Contributors: umeshsamartapu

## 👨‍💻 Author

**Umesh Samartapu**  

## 📫 Let's Connect

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/umeshsamartapu/)
[![Twitter](https://img.shields.io/badge/-Twitter-1DA1F2?style=flat-square&logo=twitter&logoColor=white)](https://x.com/umeshsamartapu)
[![Email](https://img.shields.io/badge/-Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:umeshsamartapu@gmail.com)
[![Instagram](https://img.shields.io/badge/-Instagram-E4405F?style=flat-square&logo=instagram&logoColor=white)](https://www.instagram.com/umeshsamartapu/)
[![Buy Me a Coffee](https://img.shields.io/badge/-Buy%20Me%20a%20Coffee-FBAD19?style=flat-square&logo=buymeacoffee&logoColor=black)](https://www.buymeacoffee.com/umeshsamartapu)

---

🔥 Always exploring new technologies and solving real-world problems with code!
  
