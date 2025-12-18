#!/bin/bash

echo "🔍 Running YOLOv8 Prediction on validation images..."

yolo detect predict \
  model=runs/detect/visdrone_yolov8/weights/best.pt \
  source=data/processed/yolo/images/val \
  save=True

echo "✅ Prediction completed"
