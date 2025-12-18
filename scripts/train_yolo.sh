#!/bin/bash

echo "🚀 Starting YOLOv8 Training..."

yolo detect train \
  model=yolov8n.pt \
  data=data/processed/yolo/data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=8 \
  workers=4 \
  name=visdrone_yolov8

echo "✅ Training completed"
