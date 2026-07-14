# CodeAlpha-helmet-detection-tracking-yolo


A real-time Helmet Detection and Tracking system built with **Python**, **YOLOv11**, and **OpenCV**. The application detects whether motorcycle riders are wearing helmets and supports both live webcam input and prerecorded video files.

This project was developed to demonstrate object detection, computer vision, and real-time video processing using a custom-trained YOLO model.

---

## ✨ Features

- 🎥 Real-time helmet detection using a webcam
- 📹 Process prerecorded video files
- 🪖 Detects helmet and no-helmet cases
- 📦 Powered by a custom YOLO model
- ⚡ Fast inference using Ultralytics YOLO
- 🖥️ Displays detection results in real time

---

## 🛠️ Technologies Used

- Python
- OpenCV
- Ultralytics YOLO
- PyTorch
- NumPy

---
## Screenshots

### Helmet Detected
<img width="673" height="416" alt="image" src="https://github.com/user-attachments/assets/aed7f72f-74f8-4182-9eb0-45c540d892d9" />

---
## 📁 Project Structure

```
helmet_project/
│
├── models/
│   ├── best.pt
│   └── yolo11n.pt
│
├── main.py
├── detector.py
├── tracker.py
├── video.py
├── test_model.py
├── config.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/helmet-detection.git
cd helmet-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Model

This project requires a trained YOLO model (`best.pt`).

Place the model inside:

```
models/best.pt
```

> **Note:** The trained model is **not included** in this repository due to its file size and licensing considerations.

---

## ▶️ Usage

### Webcam Detection

```bash
python main.py
```

### Detect From Video

```bash
python main.py --source path/to/video.mp4
```

Example:

```bash
python main.py --source videos/test.mp4
```

---

## 🧪 Verify the Model

Before running the application, verify that the model loads correctly:

```bash
python test_model.py
```

---

## 📸 Example Output

- Detects riders wearing helmets
- Detects riders without helmets
- Draws bounding boxes around detected objects
- Displays real-time detection confidence

---

## 📚 What I Learned

During this project, I learned:

- How YOLO object detection works
- Using custom-trained models with Ultralytics
- Real-time video processing with OpenCV
- Loading and testing deep learning models
- Working with computer vision datasets
- Debugging model loading and inference issues

---

## ⚠️ Notes

- The custom model (`best.pt`) is required for helmet detection.
- Performance depends on your hardware.
- CPU execution is supported but may be slower than GPU acceleration.

---

## 📄 License

This project is created for educational and learning purposes.
