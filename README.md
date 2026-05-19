# YOLOv12m Object Detection with Roboflow Dataset and ONNX Runtime

This repository provides an end-to-end object detection pipeline using the YOLOv12m model. It includes model training with a Roboflow dataset, ONNX model conversion, and real-time webcam inference using OpenCV and ONNX Runtime.

The main target application in this repository is card detection, but the pipeline can be adapted to other custom object detection datasets.

## Repository Overview

This project supports the following workflow:

1. Prepare a custom object detection dataset using Roboflow.
2. Train a YOLOv12m object detection model in Google Colab.
3. Validate the trained YOLO model.
4. Export the trained PyTorch model to ONNX format.
5. Run real-time object detection on a local PC using a webcam.
6. Visualize detection results using OpenCV.

## Environment
- Training Environment
- The training notebook is designed to run in Google Colab.
- Recommended environment:
  Platform: Google Colab
  GPU: NVIDIA T4, L4, A100, or equivalent
  Python: 3.10 or later
  CUDA: Provided by Google Colab runtime
  Framework: Ultralytics YOLO
  Dataset Platform: Roboflow


## Repository Structure

```text
Yolov12-ObjectDetection/
├── LICENSE
├── README.md
├── Lecture Note(3) - 오픈소스공유 & Github 활용(updated)(2026)_rev.pdf
├── Yolo12m_carddata_ivpl206.ipynb
└── Yolov12m_inference_webcam_ivpl2026.py


## Files
1. Yolo12m_carddata_ivpl206.ipynb
Google Colab notebook for YOLOv12m training using a Roboflow dataset. It includes dataset download, model training, validation, ONNX export, and optional ONNX inference testing.
2. Yolov12m_inference_webcam_ivpl2026.py
Python script for real-time webcam inference using a trained YOLOv12m ONNX model. The script uses OpenCV for video capture and visualization and ONNX Runtime for model inference.



## Main Features
1. YOLOv12m-based object detection training
2. Roboflow dataset integration
3. Google Colab training support
4. PyTorch .pt model validation
5. ONNX model export
6. ONNX Runtime inference
7. Real-time webcam object detection
8. OpenCV-based visualization
9. Custom class support
10. CPU and GPU inference support depending on ONNX Runtime configuration
