# 🌾 Crop and Weed Detection System

A robust and optimized deep learning pipeline to detect crops and weeds from images using a YOLO-formatted dataset.
This project improves upon earlier versions with bug fixes, better memory usage, and enhanced model performance.

# 🚀 About the Project
This project builds an image classification system that identifies whether an agricultural image contains only crops or crops with weeds.

The input images follow the YOLO labeling format, and the system includes:

- Data validation and visualization

- YOLO label parsing and bounding box visualization

- Automatic dataset splitting (train/validation/test)

- CNN-based classification model with Keras/TensorFlow

- Data augmentation for better generalization

- Model training, evaluation, and checkpointing

# ✨ Features
 
- 📂 Automatic loading of images and corresponding YOLO labels

- 📊 Sample visualizations with bounding boxes

- 🔥 Data augmentation: rotation, zoom, shear, flips, shifts

- 🏋️‍♂️ Custom CNN architecture for classification

- ⏳ Early stopping and model checkpointing

- 📈 Automatic plotting of class distribution and training samples

# 🏗️ Directory Structure
```bash
Code/
│
├── sample_visualizations/   # Visualized images with bounding boxes
├── predictions/             # Model prediction outputs
├── models/                  # Saved trained models
├── class_distribution.png   # Class distribution plot
├── main.py                  # Main executable Python script
├── README.md                 # Project documentation
├── requirements.txt         # Required packages
```

# 🛠️ Installation

Clone this repository and install the dependencies:
```bash
git clone https://github.com/yourusername/crop-weed-detection.git
cd crop-weed-detection
pip install -r requirements.txt
```

# ⚡ Usage
Run the pipeline with:
```bash
python main.py --data_dir PATH/TO/YOLO/DATA --classes_file PATH/TO/classes.txt
```

- --data_dir: Directory containing images and .txt YOLO label files.

- --classes_file: Text file listing class names (one per line, e.g., "crop", "weed").

Example:
```bash
python main.py --data_dir ./dataset --classes_file ./classes.txt
```

# ✅ Output:

- Trained CNN model (models/)

- Sample visualizations (sample_visualizations/)

- Class distribution plot (class_distribution.png)

- Model predictions (predictions/)

# 🏋️‍♂️ Training Details
```bash
Image Size: 512x512
Batch Size: 16
Epochs: 25
Learning Rate: 0.0001
Validation Split: 20%
Test Split: 15%
Optimizer: Adam
Loss Function: Binary Crossentropy
Metrics: Accuracy, Precision, Recall
Augmentation techniques include rotation, shift, shear, zoom, horizontal and vertical flips.
```

# 📸 Outputs

- Bounding box visualizations for 3 random samples

- Class distribution plots

- Model training logs and checkpoints

- Final trained model (.h5)

# 📦 Requirements
```bash
Python 3.8+
TensorFlow 2.x
Keras
OpenCV
Numpy
Pandas
Seaborn
Matplotlib
Pillow
Scikit-learn
(Full list in requirements.txt)
```

# 🙏 Acknowledgements

- YOLO Object Detection community

- TensorFlow and Keras teams

- Open-source datasets on agricultural weed detection

--

Made with ❤️ for advancing AI in smart agriculture 🌱

