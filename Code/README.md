# Crop and Weed Detection System 🌱

A deep learning pipeline for detecting crop and weed regions in agricultural images, based on a YOLO-formatted dataset and trained using PyTorch.

The system:

- Loads and processes labeled images

- Builds a Crop/Weed classifier using a ResNet-18 backbone

- Visualizes sample predictions

- Trains and saves the best model (best_model.pt)

# 📂 Project Structure
```bash
├── models/                 # Saved trained models (.pt files)
├── predictions/            # Prediction outputs
├── sample_visualizations/  # Sample visualizations with bounding boxes
├── crop_weed_detection.py   # Main training/testing script
├── README.md                # This file
```

# ⚙️ Requirements
```bash
Python 3.8+
PyTorch
torchvision
OpenCV
scikit-learn
matplotlib
seaborn
pandas
Pillow
```

You can install the dependencies using:
```bash
pip install torch torchvision opencv-python scikit-learn matplotlib seaborn pandas Pillow
```

# 🚀 How to Run

- Clone the repository (or copy the script).

- Prepare your dataset:

- Run the script:

```bash
python your_script.py --data_dir path_to_your_dataset
```

Available arguments:
```bash
--data_dir: Path to your dataset directory (required)

--classes: List of class names (default: ["crop", "weed"])
```

# Outputs:

- Trained model saved in /models/best_model.pt

- Sample visualizations saved in /sample_visualizations

- Predictions saved in /predictions

# 📸 Features

- Parses YOLO-format labels robustly.

- Visualizes bounding boxes with class names (Crop: Green, Weed: Red).

- Uses ResNet-18 for feature extraction and classification.

- Supports GPU (CUDA) if available.

- Saves the best model based on validation loss.

- Provides prediction functionality for single images.

# 🧠 Model

- **Backbone:** ResNet-18 (pretrained)

- **Classifier Head:** Custom fully connected layers

- **Loss Function:** Binary Cross Entropy (BCE)

- **Optimizer:** Adam

- **Scheduler:** ReduceLROnPlateau

