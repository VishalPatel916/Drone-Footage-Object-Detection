#!/bin/bash

echo "📊 Running YOLOv8 Validation..."

yolo detect val \
  model=runs/detect/visdrone_yolov8/weights/best.pt \
  data=data/processed/yolo/data.yaml

echo "✅ Validation completed"
