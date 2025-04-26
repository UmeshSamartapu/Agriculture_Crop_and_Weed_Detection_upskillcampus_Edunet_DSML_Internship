
# 🌾 Crop and Weed Detection System

An improved and optimized **Crop and Weed Detection System** built using a **YOLO-formatted dataset** and **Convolutional Neural Networks (CNN)**.  
This project focuses on classifying whether an agricultural image contains **crops only** or **crops with weeds** — helping to support smart farming and precision agriculture.

---

## 🚀 Features

- 📂 Load and preprocess YOLO-formatted image and label data.
- 🎯 Parse and visualize bounding boxes with class labels (`Crop` or `Weed`).
- 🏗️ Build a custom CNN model for binary classification (Crop-only vs Contains-Weed).
- 🔥 Data augmentation for better model generalization.
- 📊 Automatic dataset statistics and class distribution visualization.
- 🧠 Early stopping and model checkpointing for better training.
- 🖼️ Save sample visualizations and predictions.

---

## 📁 Project Structure

```
Crop-Weed-Detection/
├── sample_visualizations/    # Saved visualized images with bounding boxes
├── predictions/               # Predicted results after model testing
├── models/                    # Saved trained model files (.h5)
├── class_distribution.png     # Class distribution plot
├── README.md                  # Project documentation
├── your_script.py             # Main project code
```

---

## 🛠️ Technologies Used

- Python 3
- TensorFlow / Keras
- OpenCV
- Matplotlib & Seaborn
- Pandas
- NumPy
- scikit-learn
- PIL (Pillow)

---

## ⚙️ Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/crop-weed-detection.git
   cd crop-weed-detection
   ```

2. **Install dependencies**  
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**  
   - Organize your dataset in a folder (e.g., `data/`).
   - Each image (`.jpg`, `.jpeg`, `.png`) should have a corresponding `.txt` YOLO-format label file.
   - Create a `classes.txt` file listing class names (one per line):  
     Example:
     ```
     crop
     weed
     ```

4. **Run the system**  
   ```bash
   python your_script.py --data_dir path/to/data --classes_file path/to/classes.txt
   ```

---

## 🎨 Example Outputs

- **Sample visualization**: Images with bounding boxes drawn around detected crops and weeds.
- **Training metrics**: Accuracy, Precision, Recall.
- **Class distribution**: Bar plot showing the balance between Crop-only and Contains-Weed classes.

---

## ⚡ How it Works

- Loads images and YOLO labels.
- Parses bounding boxes, draws them on sample images.
- Builds a CNN for binary classification.
- Applies strong data augmentation during training.
- Splits data into training, validation, and test sets (stratified).
- Trains the model with early stopping and model checkpoint saving.
- Evaluates performance and saves predictions.

---

## ✨ Improvements

- ✅ Improved error handling (missing labels, corrupt images, invalid bounding boxes).
- ✅ Reduced batch size to optimize memory usage.
- ✅ Automatic sample generation and saving.
- ✅ Modular code structure for easier extension.

---

## 📢 Future Work

- [ ] Support for multi-class detection.
- [ ] Integration with real-time drone or field footage.
- [ ] Model optimization for deployment on mobile devices.

---

## 🧑‍💻 Author

- **Your Name** - [your-website.com](https://your-website.com) | [GitHub](https://github.com/your-username)

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use and modify!
