Drone vs Bird Detection Using YOLOv4
This project addresses the challenge of detecting drones in video footage, particularly distinguishing them from birds. The goal is to build a real-time drone detection model using deep learning techniques and a proprietary video dataset with annotated drone locations.

🛰️ Project Overview
With the rise in drone usage for delivery, entertainment, and surveillance, ensuring airspace security—especially around sensitive areas like airports and military bases—has become critical. This project explores a YOLOv4-based object detection pipeline trained to detect drones in videos.

Key Features
📹 Input: Drone videos with XGTF annotations

🧠 Model: YOLOv4 (fine-tuned using transfer learning)

🔁 Data Processing:

Extracted frames from videos

Converted XGTF annotations to COCO format

🔧 Techniques:

Transfer learning for faster convergence

Data augmentation for robustness

📈 Evaluation Metrics:

Precision, Recall, F1-score, mAP (mean Average Precision)

🗂️ Dataset
Format: Proprietary drone dataset (video + XGTF annotations)

Preprocessing:

Frames extracted using custom Python scripts

Annotations converted to COCO format for compatibility with YOLOv4 training pipeline

Note: Dataset cannot be shared due to proprietary restrictions.

🚀 Model Training
Pretrained YOLOv4 weights were used

Transfer learning applied for better generalization on limited data

Training and validation conducted using a COCO-style dataset

Best model selected based on weighted average of validation metrics

📊 Results
Achieved competitive performance in drone detection using limited data

Evaluated using standard object detection metrics (Precision, Recall, F1, mAP)

🛠️ Tech Stack
Python
YOLOv4 (Darknet)
OpenCV
COCO API
NumPy, Pandas

🧪 Future Work
1. Extend to differentiate between drones and birds explicitly
2. Incorporate moving camera scenarios
3. Integrate temporal tracking for more robust detections