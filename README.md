#🚁 Drone Footage Object Detection (VisDrone + YOLOv8)

Object detection system for drone imagery using the VisDrone dataset and YOLOv8.
Designed for environmental monitoring, urban planning, traffic analysis, and disaster response.

#📌 Project Overview

Dataset: VisDrone-DET

Model: YOLOv8 Nano

Framework: Ultralytics YOLO

Backend: PyTorch

Hardware: GPU (CUDA-supported)

Task: Object Detection in Drone Images

#✨ Features

YOLOv8-based object detection

Training & validation on VisDrone dataset

Metrics: Precision, Recall, mAP@50, mAP@50–95

Batch image prediction

GPU-accelerated inference

Reproducible environment

Modular training/validation/prediction scripts

#🧠 Model Details
Component	Description
Model	YOLOv8 Nano (yolov8n.pt)
Framework	Ultralytics YOLO v8.3.239
Backend	PyTorch
Classes	pedestrian, car, van, truck, bus
#📁 Project Structure
Drone-Footage-Object-Detection/
│
├── data/
│   └── processed/
│       └── yolo/
│           ├── images/
│           │   ├── train/
│           │   └── val/
│           ├── labels/
│           │   ├── train/
│           │   └── val/
│           └── data.yaml
│
├── preprocessing/
│   ├── convert_visdrone_to_yolo.py
│   └── visualize_visdrone.py
│
├── scripts/
│   ├── train_yolo.sh
│   ├── validate_yolo.sh
│   └── predict_yolo.sh
│
├── runs/
│   └── detect/
│       └── visdrone_yolov8/
│
├── requirements.txt
├── requirements-lock.txt
└── README.md

#⚙️ Environment Setup
1. Create Virtual Environment
python -m venv .venv
source .venv/bin/activate

2. Install Dependencies

Standard install

pip install -r requirements.txt


Exact reproducibility

pip install -r requirements-lock.txt

#🔁 Reproducibility

requirements.txt → high-level dependencies

requirements-lock.txt → exact versions used

Ensures consistent results across machines.

#🚀 Training
yolo detect train \
  model=yolov8n.pt \
  data=data/processed/yolo/data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=8 \
  workers=4 \
  name=visdrone_yolov8


Best model saved at:

runs/detect/visdrone_yolov8/weights/best.pt

#📊 Validation
yolo detect val \
  model=runs/detect/visdrone_yolov8/weights/best.pt \
  data=data/processed/yolo/data.yaml

Sample Results
Metric	Value
Precision	0.566
Recall	0.395
mAP@50	0.433
mAP@50–95	0.275

#🖼️ Prediction
yolo detect predict \
  model=runs/detect/visdrone_yolov8/weights/best.pt \
  source=data/processed/yolo/images/val \
  save=True


Output directory

runs/detect/predict/

#⚡ Performance

Inference time: ~4 ms/image (GPU)

Input resolution: 640 × 384

Throughput: 200+ FPS (batch inference)

#🧪 Shell Scripts
./scripts/train_yolo.sh
./scripts/validate_yolo.sh
./scripts/predict_yolo.sh


#Make executable:

chmod +x scripts/*.sh

📦 Core Dependencies

ultralytics

torch

opencv-python

numpy

pandas

matplotlib

streamlit (optional UI)

Exact versions are pinned in requirements-lock.txt.


#📚 References

Ultralytics YOLOv8: https://docs.ultralytics.com

VisDrone Dataset: http://aiskyeye.com
