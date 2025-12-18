🚁 VisDrone Object Detection using YOLOv8

This project implements object detection on the VisDrone dataset using YOLOv8 (Ultralytics).
It covers training, validation, prediction, reproducibility, and deployment-ready workflows.

📌 Features

YOLOv8-based object detection

Training on VisDrone dataset

Validation with mAP, Precision & Recall metrics

Batch prediction on images

Reproducible environment setup

GPU-accelerated (CUDA-supported)

Modular shell scripts for training, validation & prediction

🧠 Model & Framework

Model: YOLOv8 Nano (yolov8n)

Framework: Ultralytics YOLO (v8.3.239)

Backend: PyTorch

Hardware: NVIDIA GPU (CUDA enabled)

📁 Project Structure
drone_project/
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
├── README.md
└── .venv/

⚙️ Environment Setup
1️⃣ Create & Activate Virtual Environment
python -m venv .venv
source .venv/bin/activate

2️⃣ Install Dependencies
🔹 General installation
pip install -r requirements.txt

🔹 Exact reproducibility (recommended for grading/papers)
pip install -r requirements-lock.txt

🔁 Reproducibility Notes

requirements.txt → high-level dependencies

requirements-lock.txt → exact versions used during experiments

Ensures identical results across systems

🚀 Training the Model
Command
yolo detect train \
  model=yolov8n.pt \
  data=data/processed/yolo/data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=8 \
  workers=4 \
  name=visdrone_yolov8

Output

Best model saved at:

runs/detect/visdrone_yolov8/weights/best.pt


Training logs include:

Box loss

Classification loss

DFL loss

Precision, Recall, mAP@50, mAP@50-95

📊 Validation
Command
yolo detect val \
  model=runs/detect/visdrone_yolov8/weights/best.pt \
  data=data/processed/yolo/data.yaml

Metrics Reported

Precision (P)

Recall (R)

mAP@50

mAP@50-95

Per-class performance:

pedestrian

car

van

truck

bus

Example Result
mAP50-95: 0.275
Precision: 0.566
Recall: 0.395

🖼️ Prediction on Validation Images
Command
yolo detect predict \
  model=runs/detect/visdrone_yolov8/weights/best.pt \
  source=data/processed/yolo/images/val \
  save=True

Output

Annotated images saved to:

runs/detect/predict/


Console logs show per-image detections:

pedestrians, cars, vans, trucks detected

⚡ Performance

Inference Speed: ~4 ms per image (GPU)

Input Size: 640 × 384

FPS: ~200+ (batch inference)

🧪 Scripts

You can run the pipeline using shell scripts:

./scripts/train_yolo.sh
./scripts/validate_yolo.sh
./scripts/predict_yolo.sh


Make scripts executable:

chmod +x scripts/*.sh

📦 Dependencies (Core)

ultralytics==8.3.239

torch==2.9.1

opencv-python

numpy

pandas

matplotlib

streamlit (optional UI)

Full versions are pinned in requirements-lock.txt.

🔮 Future Work

Live webcam inference

Video detection

Model optimization (YOLOv8s / YOLOv8m)

Tracking (DeepSORT / ByteTrack)

Deployment using Streamlit / Flask

👤 Author

Vishal Patel
Object Detection | Computer Vision | Deep Learning

📚 References

Ultralytics YOLOv8: https://docs.ultralytics.com

VisDrone Dataset: http://aiskyeye.com/



